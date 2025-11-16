// Function: sub_8FD2C0
// Address: 0x8fd2c0
//
__int64 __fastcall sub_8FD2C0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // r15
  bool v5; // zf
  __int64 v6; // rax
  _QWORD *v7; // rax
  unsigned int v8; // eax
  __int64 v9; // rax
  _QWORD *v10; // rax
  __int64 v12; // rax
  _QWORD *v13; // rax
  char v14; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = a1 + 16;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x800000000LL;
  if ( a4 )
  {
    v5 = byte_4F6D2DC == 0;
    *(_QWORD *)(a1 + 24) = 1075836;
    *(_QWORD *)(a1 + 16) = &unk_3C92720;
    *(_QWORD *)(a1 + 32) = &unk_3C893E0;
    *(_QWORD *)(a1 + 40) = 18824;
    *(_DWORD *)(a1 + 8) = 2;
    if ( !v5 )
    {
      *(_QWORD *)(a1 + 56) = 132528;
      *(_QWORD *)(a1 + 48) = &unk_3C282A0;
      *(_QWORD *)(a1 + 64) = &unk_3C24180;
      v6 = 4;
      *(_QWORD *)(a1 + 72) = 5544;
      *(_DWORD *)(a1 + 8) = 4;
      if ( *(_DWORD *)(a1 + 12) > 4u )
        goto LABEL_7;
      goto LABEL_4;
    }
    *(_QWORD *)(a1 + 56) = 132584;
    *(_QWORD *)(a1 + 48) = &unk_3C48860;
    *(_QWORD *)(a1 + 64) = &unk_3C25740;
    *(_QWORD *)(a1 + 72) = 5544;
    *(_DWORD *)(a1 + 8) = 4;
  }
  else
  {
    *(_QWORD *)(a1 + 24) = 1074900;
    *(_QWORD *)(a1 + 16) = &unk_3D991A0;
    *(_QWORD *)(a1 + 32) = &unk_3C8DD80;
    *(_QWORD *)(a1 + 48) = &unk_3C68E60;
    *(_QWORD *)(a1 + 40) = 18824;
    *(_QWORD *)(a1 + 56) = 132464;
    *(_QWORD *)(a1 + 64) = &unk_3C26D00;
    *(_QWORD *)(a1 + 72) = 5512;
    *(_DWORD *)(a1 + 8) = 4;
  }
  v6 = 4;
  if ( *(_DWORD *)(a1 + 12) <= 4u )
  {
LABEL_4:
    sub_C8D5F0(a1, a1 + 16, 5, 16);
    v6 = *(unsigned int *)(a1 + 8);
  }
LABEL_7:
  v7 = (_QWORD *)(*(_QWORD *)a1 + 16 * v6);
  *v7 = &unk_4B7FEE0;
  v7[1] = 69872;
  ++*(_DWORD *)(a1 + 8);
  v15[0] = &v14;
  *(_QWORD *)(__readfsqword(0) - 24) = v15;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_8FC3F0;
  if ( !&_pthread_key_create )
  {
    v8 = -1;
    goto LABEL_19;
  }
  v8 = pthread_once(&once_control, init_routine);
  if ( v8 )
    goto LABEL_19;
  v9 = *(unsigned int *)(a1 + 8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
  {
    sub_C8D5F0(a1, v4, v9 + 1, 16);
    v9 = *(unsigned int *)(a1 + 8);
  }
  v10 = (_QWORD *)(*(_QWORD *)a1 + 16 * v9);
  *v10 = &unk_4B7F680;
  v10[1] = 2144;
  ++*(_DWORD *)(a1 + 8);
  if ( byte_4F6D2D0 )
  {
    v15[0] = &v14;
    *(_QWORD *)(__readfsqword(0) - 24) = v15;
    *(_QWORD *)(__readfsqword(0) - 32) = sub_8FC410;
    v8 = pthread_once(&dword_4F6D2D4, init_routine);
    if ( !v8 )
    {
      v12 = *(unsigned int *)(a1 + 8);
      if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        sub_C8D5F0(a1, v4, v12 + 1, 16);
        v12 = *(unsigned int *)(a1 + 8);
      }
      v13 = (_QWORD *)(*(_QWORD *)a1 + 16 * v12);
      *v13 = &unk_4B7F2A0;
      v13[1] = 984;
      ++*(_DWORD *)(a1 + 8);
      return a1;
    }
LABEL_19:
    sub_4264C5(v8);
  }
  return a1;
}
