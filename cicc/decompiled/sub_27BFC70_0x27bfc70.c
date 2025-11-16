// Function: sub_27BFC70
// Address: 0x27bfc70
//
__int64 __fastcall sub_27BFC70(void **a1, _QWORD *a2)
{
  int v2; // eax
  unsigned int v3; // r8d
  unsigned __int8 v5; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = 0;
  if ( *a1 == sub_C33340() )
    v2 = sub_C3FF40((__int64)a1, v6, 1, 0x40u, 1u, 0, &v5);
  else
    v2 = sub_C34710((__int64)a1, v6, 1, 0x40u, 1u, 0, &v5);
  v3 = 0;
  if ( !v2 )
  {
    v3 = v5;
    if ( v5 )
      *a2 = v6[0];
  }
  return v3;
}
