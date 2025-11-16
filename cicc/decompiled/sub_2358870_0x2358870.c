// Function: sub_2358870
// Address: 0x2358870
//
void __fastcall sub_2358870(unsigned __int64 *a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 *v3; // r12
  unsigned __int64 v4; // r14
  _QWORD *v5; // rax
  unsigned __int64 v6; // rdi
  unsigned __int64 *v7; // rbx
  unsigned __int64 v8[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *a2;
  v3 = (unsigned __int64 *)a2[1];
  *a2 = 0;
  v4 = a2[2];
  a2[1] = 0;
  a2[2] = 0;
  v5 = (_QWORD *)sub_22077B0(0x20u);
  if ( v5 )
  {
    v5[1] = v2;
    v5[2] = v3;
    v2 = 0;
    v3 = 0;
    v5[3] = v4;
    *v5 = &unk_4A0D178;
    v8[0] = (unsigned __int64)v5;
    sub_2356EF0(a1, v8);
    v6 = v8[0];
    if ( !v8[0] )
      return;
  }
  else
  {
    v8[0] = 0;
    sub_2356EF0(a1, v8);
    v6 = v8[0];
    if ( !v8[0] )
      goto LABEL_4;
  }
  (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v6 + 8LL))(v6);
LABEL_4:
  if ( v3 != (unsigned __int64 *)v2 )
  {
    v7 = (unsigned __int64 *)v2;
    do
    {
      if ( (unsigned __int64 *)*v7 != v7 + 2 )
        j_j___libc_free_0(*v7);
      v7 += 4;
    }
    while ( v7 != v3 );
  }
  if ( v2 )
    j_j___libc_free_0(v2);
}
