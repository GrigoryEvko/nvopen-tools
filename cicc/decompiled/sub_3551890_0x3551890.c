// Function: sub_3551890
// Address: 0x3551890
//
__int64 __fastcall sub_3551890(__int64 a1, __int64 a2)
{
  unsigned int v3; // r14d
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v7; // r15
  _QWORD *(__fastcall *v8)(_QWORD *); // r14
  _QWORD *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rax
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned int)((__int64)(*(_QWORD *)(a2 + 40) - *(_QWORD *)(a2 + 32)) >> 3) != 1 )
  {
    v3 = 0;
    sub_3548560(*(__int64 ***)(a1 + 208), a2);
    return v3;
  }
  v3 = *(unsigned __int8 *)(a1 + 568);
  if ( (_BYTE)v3 )
  {
    v3 = 0;
    sub_3548870(*(__int64 ***)(a1 + 208), a2);
    return v3;
  }
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  v4 = *(_QWORD *)(a1 + 240);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 344LL);
  if ( v5 == sub_2DB1AE0
    || ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, __int64, __int64, _QWORD))v5)(
         v4,
         **(_QWORD **)(a2 + 32),
         a1 + 576,
         a1 + 584,
         a1 + 592,
         0) )
  {
    sub_3549020(*(__int64 ***)(a1 + 208), a2);
    return v3;
  }
  v7 = *(_QWORD *)(a1 + 240);
  *(_QWORD *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 776) = 0;
  v8 = *(_QWORD *(__fastcall **)(_QWORD *))(*(_QWORD *)v7 + 376LL);
  v9 = sub_2EA6400(a2);
  if ( v8 == sub_2FDC520 )
  {
    v10 = *(_QWORD *)(a1 + 784);
    v12[0] = 0;
    *(_QWORD *)(a1 + 784) = 0;
    if ( !v10 )
    {
LABEL_19:
      v3 = 0;
      sub_3548B00(*(__int64 ***)(a1 + 208), a2);
      return v3;
    }
  }
  else
  {
    ((void (__fastcall *)(_QWORD *, __int64, _QWORD *))v8)(v12, v7, v9);
    v11 = v12[0];
    v10 = *(_QWORD *)(a1 + 784);
    v12[0] = 0;
    *(_QWORD *)(a1 + 784) = v11;
    if ( !v10 )
      goto LABEL_13;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
  if ( v12[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v12[0] + 8LL))(v12[0]);
  v11 = *(_QWORD *)(a1 + 784);
LABEL_13:
  if ( !v11 )
    goto LABEL_19;
  if ( sub_2EA49A0(a2) )
  {
    v3 = 1;
    sub_35512F0((_QWORD *)a1, **(_QWORD **)(a2 + 32));
  }
  else
  {
    v3 = 0;
    sub_3548D90(*(__int64 ***)(a1 + 208), a2);
  }
  return v3;
}
