// Function: sub_22EC7B0
// Address: 0x22ec7b0
//
__int64 __fastcall sub_22EC7B0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // [rsp+8h] [rbp-48h]
  _QWORD v11[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !*a4 )
    return sub_3140C00(a1);
  (*(void (__fastcall **)(_QWORD *))(*(_QWORD *)*a4 + 16LL))(v11);
  if ( v11[0] )
  {
    if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 24LL))(v11[0]) == &unk_4C5D162 )
    {
      v10 = *(_QWORD *)(v11[0] + 8LL);
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
      if ( v10 )
      {
        sub_22EC410(a1, a2, a3, v10);
        return sub_3140C00(a1);
      }
    }
    else if ( v11[0] )
    {
      (*(void (**)(void))(*(_QWORD *)v11[0] + 8LL))();
      v8 = *a4;
      goto LABEL_8;
    }
  }
  v8 = *a4;
LABEL_8:
  if ( v8 )
  {
    (*(void (__fastcall **)(_QWORD *))(*(_QWORD *)v8 + 16LL))(v11);
    if ( v11[0] )
    {
      if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 24LL))(v11[0]) == &unk_4C5D161 )
      {
        v9 = *(_QWORD *)(v11[0] + 8LL);
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
        if ( v9 )
          sub_22EC6D0(a1, a2, a3, v9);
      }
      else if ( v11[0] )
      {
        (*(void (**)(void))(*(_QWORD *)v11[0] + 8LL))();
      }
    }
  }
  return sub_3140C00(a1);
}
