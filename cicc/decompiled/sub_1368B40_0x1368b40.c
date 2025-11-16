// Function: sub_1368B40
// Address: 0x1368b40
//
__int64 __fastcall sub_1368B40(__int64 *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdi
  int v4; // edx
  int v5; // ecx
  __int64 v6; // r8
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  int v11; // eax
  int v12; // r10d
  int v13; // [rsp+Ch] [rbp-4h] BYREF

  v2 = -1;
  v3 = *a1;
  v4 = *(_DWORD *)(v3 + 184);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = *(_QWORD *)(v3 + 168);
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      v2 = *((_DWORD *)v8 + 2);
    }
    else
    {
      v11 = 1;
      while ( v9 != -8 )
      {
        v12 = v11 + 1;
        v7 = v5 & (v11 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v11 = v12;
      }
      v2 = -1;
    }
  }
  v13 = v2;
  return sub_1370EA0(v3, &v13);
}
