// Function: sub_FFBA00
// Address: 0xffba00
//
__int64 __fastcall sub_FFBA00(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 result; // rax
  char v4; // dl
  _QWORD **v5; // rax
  _QWORD **v6; // rbx
  _QWORD *v7; // r12
  _QWORD **v8; // r14
  __int64 v9; // r13
  _QWORD **v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rbx
  _QWORD *v14; // r14
  void (__fastcall *v15)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v16; // rax
  _QWORD *v17; // [rsp-40h] [rbp-40h]

  v2 = *(unsigned int *)(a1 + 588);
  result = 0;
  if ( *(_DWORD *)(a1 + 592) == (_DWORD)v2 )
    return result;
  v4 = *(_BYTE *)(a1 + 596);
  v5 = *(_QWORD ***)(a1 + 576);
  if ( !v4 )
    v2 = *(unsigned int *)(a1 + 584);
  v6 = &v5[v2];
  if ( v5 == v6 )
  {
LABEL_7:
    v9 = a1 + 568;
  }
  else
  {
    while ( 1 )
    {
      v7 = *v5;
      v8 = v5;
      if ( (unsigned __int64)*v5 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v6 == ++v5 )
        goto LABEL_7;
    }
    v9 = a1 + 568;
    if ( v6 != v5 )
    {
      do
      {
        a2 = (__int64)v7;
        sub_FFB730(a1, (__int64)v7);
        sub_AA5450(v7);
        v10 = v8 + 1;
        if ( v8 + 1 == v6 )
          break;
        while ( 1 )
        {
          v7 = *v10;
          v8 = v10;
          if ( (unsigned __int64)*v10 < 0xFFFFFFFFFFFFFFFELL )
            break;
          if ( v6 == ++v10 )
            goto LABEL_12;
        }
      }
      while ( v6 != v10 );
LABEL_12:
      v4 = *(_BYTE *)(a1 + 596);
    }
  }
  ++*(_QWORD *)(a1 + 568);
  if ( v4 )
    goto LABEL_18;
  v11 = 4 * (*(_DWORD *)(a1 + 588) - *(_DWORD *)(a1 + 592));
  v12 = *(unsigned int *)(a1 + 584);
  if ( v11 < 0x20 )
    v11 = 32;
  if ( (unsigned int)v12 <= v11 )
  {
    memset(*(void **)(a1 + 576), -1, 8 * v12);
LABEL_18:
    *(_QWORD *)(a1 + 588) = 0;
    goto LABEL_19;
  }
  sub_C8C990(v9, a2);
LABEL_19:
  v13 = *(_QWORD **)(a1 + 680);
  v17 = *(_QWORD **)(a1 + 672);
  result = 1;
  if ( v17 != v13 )
  {
    v14 = *(_QWORD **)(a1 + 672);
    do
    {
      v15 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v14[7];
      *v14 = &unk_49E5048;
      if ( v15 )
        v15(v14 + 5, v14 + 5, 3);
      *v14 = &unk_49DB368;
      v16 = v14[3];
      if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
        sub_BD60C0(v14 + 1);
      v14 += 9;
    }
    while ( v13 != v14 );
    *(_QWORD *)(a1 + 680) = v17;
    return 1;
  }
  return result;
}
