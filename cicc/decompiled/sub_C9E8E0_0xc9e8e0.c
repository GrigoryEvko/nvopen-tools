// Function: sub_C9E8E0
// Address: 0xc9e8e0
//
int __fastcall sub_C9E8E0(__int64 *a1, _BYTE *a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  __int64 v7; // [rsp+0h] [rbp-30h]
  _BYTE *v8; // [rsp+8h] [rbp-28h]

  if ( !qword_4F84F60 )
  {
    v7 = a5;
    v8 = a4;
    sub_C7D570(&qword_4F84F60, sub_CA0780, (__int64)sub_C9FD10);
    a5 = v7;
    a4 = v8;
  }
  return sub_C9E810(a1, a2, a3, a4, a5, (pthread_mutex_t *)(qword_4F84F60 + 664));
}
