// Function: sub_CC78E0
// Address: 0xcc78e0
//
unsigned __int64 __fastcall sub_CC78E0(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rbx
  char *v4; // r14
  unsigned __int64 v5; // r15
  char *v6; // rax
  size_t v7; // rdx
  char *v8; // r8
  size_t v9; // r12
  int v10; // eax
  unsigned __int8 v11; // dl
  unsigned int v12; // r10d
  __int64 v13; // rdi
  __int64 v14; // r9
  __int64 v16; // [rsp+8h] [rbp-48h]
  __int128 v17; // [rsp+10h] [rbp-40h] BYREF

  v1 = sub_CC7380((__int64 *)a1);
  v3 = v2;
  v16 = v1;
  v4 = (char *)v1;
  v5 = v2;
  v6 = sub_CC6100(*(_DWORD *)(a1 + 44));
  v8 = (char *)v16;
  if ( v3 < v7 || (v9 = v7) != 0 && (v10 = memcmp(v4, v6, v7), v8 = (char *)v16, v10) )
  {
    if ( *(_DWORD *)(a1 + 44) == 9 )
    {
      if ( v3 > 4 && *(_DWORD *)v4 == 1868783981 && v4[4] == 115 )
      {
        v5 = v3 - 5;
        v8 = v4 + 5;
      }
    }
    else if ( v3 > 7 && *(_QWORD *)v4 == 0x736F6E6F69736976LL )
    {
      v5 = v3 - 8;
      v8 = v4 + 8;
    }
  }
  else
  {
    v8 = &v4[v9];
    v5 = v3 - v9;
  }
  v17 = 0;
  sub_F05080(&v17, v8, v5);
  if ( v17 < 0 )
  {
    v13 = (unsigned int)v17;
    v11 = 1;
    v14 = DWORD2(v17) & 0x7FFFFFFF;
    v12 = DWORD1(v17) & 0x7FFFFFFF;
  }
  else
  {
    v11 = BYTE7(v17) >> 7;
    v12 = DWORD1(v17) & 0x7FFFFFFF;
    v13 = (unsigned int)v17;
    v14 = DWORD2(v17) & 0x7FFFFFFF;
  }
  return v13 | (((v14 << 32) | v12 | ((unsigned __int64)v11 << 31)) << 32);
}
