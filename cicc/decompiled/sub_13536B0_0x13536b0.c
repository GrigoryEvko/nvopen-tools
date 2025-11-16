// Function: sub_13536B0
// Address: 0x13536b0
//
__int64 __fastcall sub_13536B0(__int64 a1, const char *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r12

  v2 = (_QWORD *)sub_22077B0(104);
  if ( v2 )
  {
    *v2 = 0;
    v2[1] = 0;
    v2[2] = 0;
    v2[3] = 0;
    v2[4] = 0;
    v2[5] = 0;
    v2[6] = 0;
    v2[7] = 0;
    v2[8] = 0;
    v2[9] = 0;
    v2[10] = 0;
    v2[11] = 0;
    v2[12] = 0;
  }
  v4 = *(_QWORD **)(a1 + 160);
  *(_QWORD *)(a1 + 160) = v2;
  if ( v4 )
  {
    sub_13525A0(v4, a2, v3);
    j_j___libc_free_0(v4, 104);
  }
  return 0;
}
