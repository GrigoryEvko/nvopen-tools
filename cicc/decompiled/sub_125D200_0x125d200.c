// Function: sub_125D200
// Address: 0x125d200
//
__int64 __fastcall sub_125D200(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r15
  bool v5; // zf
  unsigned int v6; // eax
  __int64 v7; // rax
  _QWORD *v8; // rax
  __int64 v10; // rax
  _QWORD *v11; // rax
  char v12; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v13[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a1 + 16;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  if ( a4 )
  {
    v5 = byte_4F92C5C == 0;
    *(_QWORD *)(a1 + 24) = 1075836;
    *(_QWORD *)(a1 + 16) = &unk_4002C00;
    *(_DWORD *)(a1 + 8) = 1;
    if ( v5 )
    {
      *(_QWORD *)(a1 + 40) = 132584;
      *(_QWORD *)(a1 + 32) = &unk_3FC2080;
      *(_QWORD *)(a1 + 48) = &unk_3F9EF60;
    }
    else
    {
      *(_QWORD *)(a1 + 40) = 132528;
      *(_QWORD *)(a1 + 32) = &unk_3FA1AC0;
      *(_QWORD *)(a1 + 48) = &unk_3F9D9A0;
    }
    *(_QWORD *)(a1 + 56) = 5544;
    *(_DWORD *)(a1 + 8) = 3;
  }
  else
  {
    *(_QWORD *)(a1 + 24) = 1074900;
    *(_QWORD *)(a1 + 16) = &unk_4109680;
    *(_QWORD *)(a1 + 32) = &unk_3FE2680;
    *(_QWORD *)(a1 + 40) = 132464;
    *(_QWORD *)(a1 + 48) = &unk_3FA0520;
    *(_QWORD *)(a1 + 56) = 5512;
    *(_DWORD *)(a1 + 8) = 3;
  }
  ++*(_DWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 64) = &unk_4C5DDE0;
  *(_QWORD *)(a1 + 72) = 69872;
  v13[0] = &v12;
  *(_QWORD *)(__readfsqword(0) - 24) = v13;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_125C330;
  if ( !&_pthread_key_create )
  {
    v6 = -1;
    goto LABEL_18;
  }
  v6 = pthread_once(&dword_4F92C58, init_routine);
  if ( v6 )
    goto LABEL_18;
  v7 = *(unsigned int *)(a1 + 8);
  if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, v4, 0, 16);
    v7 = *(unsigned int *)(a1 + 8);
  }
  v8 = (_QWORD *)(*(_QWORD *)a1 + 16 * v7);
  *v8 = &unk_4C5D580;
  v8[1] = 2144;
  ++*(_DWORD *)(a1 + 8);
  if ( byte_4F92C50 )
  {
    v13[0] = &v12;
    *(_QWORD *)(__readfsqword(0) - 24) = v13;
    *(_QWORD *)(__readfsqword(0) - 32) = sub_125C350;
    v6 = pthread_once(&dword_4F92C54, init_routine);
    if ( !v6 )
    {
      v10 = *(unsigned int *)(a1 + 8);
      if ( (unsigned int)v10 >= *(_DWORD *)(a1 + 12) )
      {
        sub_16CD150(a1, v4, 0, 16);
        v10 = *(unsigned int *)(a1 + 8);
      }
      v11 = (_QWORD *)(*(_QWORD *)a1 + 16 * v10);
      *v11 = &unk_4C5D1A0;
      v11[1] = 984;
      ++*(_DWORD *)(a1 + 8);
      return a1;
    }
LABEL_18:
    sub_4264C5(v6);
  }
  return a1;
}
