// Function: sub_2358750
// Address: 0x2358750
//
void __fastcall sub_2358750(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  unsigned __int64 *v3; // r12
  __int64 v4; // r14
  char v5; // bl
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)a2;
  v3 = *(unsigned __int64 **)(a2 + 8);
  *(_QWORD *)a2 = 0;
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_BYTE *)(a2 + 24);
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 8) = 0;
  v6 = sub_22077B0(0x28u);
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = v2;
    *(_QWORD *)(v6 + 16) = v3;
    v2 = 0;
    v3 = 0;
    *(_QWORD *)(v6 + 24) = v4;
    *(_QWORD *)v6 = &unk_4A0D278;
    *(_BYTE *)(v6 + 32) = v5;
    v9[0] = v6;
    sub_2356EF0(a1, v9);
    v7 = v9[0];
    if ( !v9[0] )
      return;
  }
  else
  {
    v9[0] = 0;
    sub_2356EF0(a1, v9);
    v7 = v9[0];
    if ( !v9[0] )
      goto LABEL_4;
  }
  (*(void (__fastcall **)(unsigned __int64))(*(_QWORD *)v7 + 8LL))(v7);
LABEL_4:
  if ( v3 != (unsigned __int64 *)v2 )
  {
    v8 = (unsigned __int64 *)v2;
    do
    {
      if ( *v8 )
        j_j___libc_free_0(*v8);
      v8 += 3;
    }
    while ( v8 != v3 );
  }
  if ( v2 )
    j_j___libc_free_0(v2);
}
