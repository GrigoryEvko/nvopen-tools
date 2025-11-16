// Function: sub_1DDC3C0
// Address: 0x1ddc3c0
//
__int64 __fastcall sub_1DDC3C0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // rdi
  unsigned int v4; // eax
  int v5; // edx
  int v6; // ecx
  __int64 v7; // r8
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r9
  int v12; // eax
  int v13; // r10d
  unsigned int v14; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v15; // [rsp-8h] [rbp-8h]

  v3 = *(_QWORD *)(a1 + 232);
  if ( !v3 )
    return 0;
  v15 = v2;
  v4 = -1;
  v5 = *(_DWORD *)(v3 + 184);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = *(_QWORD *)(v3 + 168);
    v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_4:
      v4 = *((_DWORD *)v9 + 2);
    }
    else
    {
      v12 = 1;
      while ( v10 != -8 )
      {
        v13 = v12 + 1;
        v8 = v6 & (v12 + v8);
        v9 = (__int64 *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( a2 == *v9 )
          goto LABEL_4;
        v12 = v13;
      }
      v4 = -1;
    }
  }
  v14 = v4;
  return sub_1370CD0(v3, &v14);
}
