// Function: sub_23A2270
// Address: 0x23a2270
//
void __fastcall sub_23A2270(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdi
  _QWORD *v7; // rbx
  unsigned __int64 v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *a2;
  v3 = (_QWORD *)a2[1];
  *a2 = 0;
  v4 = a2[2];
  a2[1] = 0;
  a2[2] = 0;
  v5 = (_QWORD *)sub_22077B0(0x30u);
  if ( v5 )
  {
    v5[1] = v2;
    v5[2] = v3;
    v2 = 0;
    v3 = 0;
    v5[3] = v4;
    *v5 = &unk_4A0C378;
    v5[4] = 0;
    v5[5] = 0;
    v8[0] = (unsigned __int64)v5;
    sub_23A2230(a1, v8);
    v6 = v8[0];
    if ( !v8[0] )
      return;
  }
  else
  {
    v8[0] = 0;
    sub_23A2230(a1, v8);
    v6 = v8[0];
    if ( !v8[0] )
      goto LABEL_4;
  }
  (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v6 + 8LL))(v6);
LABEL_4:
  if ( v3 != (_QWORD *)v2 )
  {
    v7 = (_QWORD *)v2;
    do
    {
      if ( *v7 )
        (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*v7 + 8LL))(*v7);
      ++v7;
    }
    while ( v7 != v3 );
  }
  if ( v2 )
    j_j___libc_free_0(v2);
}
