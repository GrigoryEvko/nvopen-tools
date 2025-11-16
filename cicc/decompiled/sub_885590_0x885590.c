// Function: sub_885590
// Address: 0x885590
//
void __fastcall sub_885590(__int64 *a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  _QWORD **v3; // r12
  _QWORD *v4; // rax
  _QWORD v5[5]; // [rsp-28h] [rbp-28h] BYREF

  if ( a2 )
  {
    v2 = *a1;
    v5[1] = 0;
    v5[0] = v2;
    v3 = (_QWORD **)sub_881B20(a2, (__int64)v5, 1);
    v4 = *v3;
    if ( !*v3 )
    {
      v4 = (_QWORD *)sub_823970(16);
      *v4 = 0;
      v4[1] = 0;
      *v4 = *a1;
      *v3 = v4;
    }
    a1[4] = v4[1];
    v4[1] = a1;
  }
}
