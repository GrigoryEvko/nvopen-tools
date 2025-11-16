// Function: sub_22EC4E0
// Address: 0x22ec4e0
//
void *__fastcall sub_22EC4E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  void *result; // rax
  __int64 v8; // rsi
  __int64 v9; // r15
  __int64 v10; // [rsp+8h] [rbp-48h]
  _QWORD v11[7]; // [rsp+18h] [rbp-38h] BYREF

  result = (void *)sub_31413C0();
  if ( !*a4 )
    return result;
  result = (void *)(*(__int64 (__fastcall **)(_QWORD *))(*(_QWORD *)*a4 + 16LL))(v11);
  if ( v11[0] )
  {
    result = (void *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 24LL))(v11[0]);
    if ( result == &unk_4C5D162 )
    {
      v10 = *(_QWORD *)(v11[0] + 8LL);
      result = (void *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
      if ( v10 )
        return (void *)sub_22EC3A0(a1, a2, a3, v10, 1u);
    }
    else if ( v11[0] )
    {
      result = (void *)(*(__int64 (**)(void))(*(_QWORD *)v11[0] + 8LL))();
      v8 = *a4;
      goto LABEL_7;
    }
  }
  v8 = *a4;
LABEL_7:
  if ( v8 )
  {
    result = (void *)(*(__int64 (__fastcall **)(_QWORD *))(*(_QWORD *)v8 + 16LL))(v11);
    if ( v11[0] )
    {
      result = (void *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 24LL))(v11[0]);
      if ( result == &unk_4C5D161 )
      {
        v9 = *(_QWORD *)(v11[0] + 8LL);
        result = (void *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
        if ( v9 )
          return (void *)sub_22EC110(a1, a2, a3, v9, 1u);
      }
      else if ( v11[0] )
      {
        return (void *)(*(__int64 (**)(void))(*(_QWORD *)v11[0] + 8LL))();
      }
    }
  }
  return result;
}
