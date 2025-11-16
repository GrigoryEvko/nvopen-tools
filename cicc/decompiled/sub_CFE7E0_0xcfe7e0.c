// Function: sub_CFE7E0
// Address: 0xcfe7e0
//
__int64 __fastcall sub_CFE7E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  void (*v6)(void); // rax
  __int64 v7; // r13
  unsigned int v8; // r14d
  __int64 result; // rax
  char v10; // cl
  _QWORD *v11; // r12
  unsigned int v12; // r13d
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 i; // rdx
  char v16; // cl
  _QWORD *v17; // rbx
  char v18; // al
  __int64 v19; // rax
  bool v20; // zf
  _QWORD v21[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v22; // [rsp+18h] [rbp-38h]
  __int64 v23; // [rsp+20h] [rbp-30h]

  v6 = *(void (**)(void))(*(_QWORD *)a1 + 128LL);
  if ( (char *)v6 == (char *)sub_CFE780 )
  {
    if ( (_BYTE)qword_4F866C8 )
      sub_CFE4C0(a1, a2, (__int64)sub_CFE780, a4, a5, a6);
  }
  else
  {
    v6();
  }
  v7 = *(unsigned int *)(a1 + 192);
  v8 = *(_DWORD *)(a1 + 200);
  result = sub_CFBE20(a1 + 176, a2);
  if ( (_DWORD)v7 )
  {
    result = (unsigned int)(v7 - 1);
    v7 = 64;
    if ( (_DWORD)result )
    {
      _BitScanReverse((unsigned int *)&result, result);
      v10 = 33 - (result ^ 0x1F);
      result = 64;
      v7 = (unsigned int)(1 << v10);
      if ( (int)v7 < 64 )
        v7 = 64;
    }
  }
  v11 = *(_QWORD **)(a1 + 184);
  if ( *(_DWORD *)(a1 + 200) == (_DWORD)v7 )
  {
    v21[0] = 2;
    *(_QWORD *)(a1 + 192) = 0;
    v21[1] = 0;
    v17 = &v11[6 * v7];
    v22 = -4096;
    v23 = 0;
    if ( v17 != v11 )
    {
      do
      {
        if ( v11 )
        {
          v18 = v21[0];
          v11[2] = 0;
          v11[1] = v18 & 6;
          v19 = v22;
          v20 = v22 == 0;
          v11[3] = v22;
          if ( v19 != -4096 && !v20 && v19 != -8192 )
            sub_BD6050(v11 + 1, v21[0] & 0xFFFFFFFFFFFFFFF8LL);
          *v11 = &unk_49DDB10;
          v11[4] = v23;
        }
        v11 += 6;
      }
      while ( v17 != v11 );
      result = v22;
      if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
        return sub_BD60C0(v21);
    }
  }
  else
  {
    result = sub_C7D6A0(*(_QWORD *)(a1 + 184), 48LL * v8, 8);
    if ( (_DWORD)v7 )
    {
      v12 = 4 * (int)v7 / 3u;
      v13 = ((((((((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
                 | (v12 + 1)
                 | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
               | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
               | (v12 + 1)
               | ((unsigned __int64)(v12 + 1) >> 1)) >> 8)
             | (((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2) | (v12 + 1)
                                                                         | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
             | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
             | (v12 + 1)
             | ((unsigned __int64)(v12 + 1) >> 1)) >> 16)
           | (((((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2) | (v12 + 1)
                                                                         | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
             | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
             | (v12 + 1)
             | ((unsigned __int64)(v12 + 1) >> 1)) >> 8)
           | (((((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2) | (v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 4)
           | (((v12 + 1) | ((unsigned __int64)(v12 + 1) >> 1)) >> 2)
           | (v12 + 1)
           | ((unsigned __int64)(v12 + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 200) = v13;
      result = sub_C7D670(48 * v13, 8);
      *(_QWORD *)(a1 + 192) = 0;
      *(_QWORD *)(a1 + 184) = result;
      v14 = *(unsigned int *)(a1 + 200);
      v21[0] = 2;
      v23 = 0;
      for ( i = result + 48 * v14; i != result; result += 48 )
      {
        if ( result )
        {
          v16 = v21[0];
          *(_QWORD *)(result + 16) = 0;
          *(_QWORD *)(result + 24) = -4096;
          *(_QWORD *)result = &unk_49DDB10;
          *(_QWORD *)(result + 8) = v16 & 6;
          *(_QWORD *)(result + 32) = v23;
        }
      }
    }
    else
    {
      *(_QWORD *)(a1 + 184) = 0;
      *(_QWORD *)(a1 + 192) = 0;
      *(_DWORD *)(a1 + 200) = 0;
    }
  }
  return result;
}
