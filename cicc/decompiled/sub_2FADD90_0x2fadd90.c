// Function: sub_2FADD90
// Address: 0x2fadd90
//
void __fastcall sub_2FADD90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdx
  __int64 v4; // r8
  unsigned int v5; // ecx
  __int64 *v6; // rax
  __int64 v7; // r10
  __int64 v8; // rdx
  unsigned __int64 v9; // rcx
  int v10; // eax
  __int64 v11; // rax
  int v12; // r11d
  __int64 v13; // [rsp-48h] [rbp-48h] BYREF
  __int64 v14; // [rsp-40h] [rbp-40h] BYREF
  _QWORD v15[7]; // [rsp-38h] [rbp-38h] BYREF

  v3 = *(unsigned int *)(a1 + 144);
  v4 = *(_QWORD *)(a1 + 128);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
LABEL_3:
      if ( v6 != (__int64 *)(v4 + 16 * v3) )
      {
        v8 = v6[1];
        *v6 = -8192;
        --*(_DWORD *)(a1 + 136);
        ++*(_DWORD *)(a1 + 140);
        v9 = v8 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(a2 + 44) & 8) != 0 )
        {
          v15[6] = v2;
          v11 = *(_QWORD *)(a2 + 8);
          v14 = v8;
          *(_QWORD *)(v9 + 16) = v11;
          v13 = v11;
          sub_2FADAE0((__int64)v15, a1 + 120, &v13, &v14);
        }
        else
        {
          *(_QWORD *)(v9 + 16) = 0;
        }
      }
    }
    else
    {
      v10 = 1;
      while ( v7 != -4096 )
      {
        v12 = v10 + 1;
        v5 = (v3 - 1) & (v10 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          goto LABEL_3;
        v10 = v12;
      }
    }
  }
}
