// Function: sub_D4BD90
// Address: 0xd4bd90
//
void __fastcall sub_D4BD90(__int64 a1, char *a2, __int64 a3, __m128i a4)
{
  char v7; // al
  size_t v8; // rdx
  unsigned __int8 *v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 *v12; // r14
  __int64 *i; // rbx
  const char *v14; // rsi
  __int64 *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 (__fastcall **v18)(); // rax
  __int64 *v19; // r13
  __int64 *v20; // rbx
  __int64 *v21; // [rsp+0h] [rbp-80h] BYREF
  __int64 v22; // [rsp+8h] [rbp-78h]
  _BYTE v23[112]; // [rsp+10h] [rbp-70h] BYREF

  if ( (unsigned __int8)sub_BC5DE0() )
  {
    v17 = sub_CB6200((__int64)a2, *(unsigned __int8 **)a3, *(_QWORD *)(a3 + 8));
    sub_904010(v17, " (loop: ");
    sub_A5BF40(**(unsigned __int8 ***)(a1 + 32), (__int64)a2, 0, 0);
    sub_904010((__int64)a2, ")\n");
    v18 = (__int64 (__fastcall **)())sub_AA4B30(**(_QWORD **)(a1 + 32));
    sub_A69980(v18, (__int64)a2, 0, 0, 0, a4);
    return;
  }
  if ( (unsigned __int8)sub_BC5DF0() )
  {
    v8 = *(_QWORD *)(a3 + 8);
    v9 = *(unsigned __int8 **)a3;
    v10 = (__int64)a2;
    goto LABEL_17;
  }
  v7 = sub_BC5DF0();
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(unsigned __int8 **)a3;
  v10 = (__int64)a2;
  if ( v7 )
  {
LABEL_17:
    v16 = sub_CB6200(v10, v9, v8);
    sub_904010(v16, " (loop: ");
    sub_A5BF40(**(unsigned __int8 ***)(a1 + 32), (__int64)a2, 0, 0);
    sub_904010((__int64)a2, ")\n");
    sub_A69870(*(_QWORD *)(**(_QWORD **)(a1 + 32) + 72LL), a2, 0);
    return;
  }
  sub_CB6200((__int64)a2, v9, v8);
  v11 = sub_D4B130(a1);
  if ( v11 )
  {
    sub_904010((__int64)a2, "\n; Preheader:");
    sub_A68DD0(v11, (__int64)a2, 0, 0, 0);
    sub_904010((__int64)a2, "\n; Loop:");
  }
  v12 = *(__int64 **)(a1 + 40);
  for ( i = *(__int64 **)(a1 + 32); v12 != i; ++i )
  {
    while ( *i )
    {
      sub_A68DD0(*i++, (__int64)a2, 0, 0, 0);
      if ( v12 == i )
        goto LABEL_11;
    }
    sub_904010((__int64)a2, "Printing <null> block");
  }
LABEL_11:
  v14 = (const char *)&v21;
  v22 = 0x800000000LL;
  v21 = (__int64 *)v23;
  sub_D472F0(a1, (__int64)&v21);
  if ( !(_DWORD)v22 )
    goto LABEL_12;
  v14 = "\n; Exit blocks";
  sub_904010((__int64)a2, "\n; Exit blocks");
  v15 = v21;
  v19 = &v21[(unsigned int)v22];
  if ( v19 != v21 )
  {
    v20 = v21;
    do
    {
      if ( *v20 )
      {
        v14 = a2;
        sub_A68DD0(*v20, (__int64)a2, 0, 0, 0);
      }
      else
      {
        v14 = "Printing <null> block";
        sub_904010((__int64)a2, "Printing <null> block");
      }
      ++v20;
    }
    while ( v19 != v20 );
LABEL_12:
    v15 = v21;
  }
  if ( v15 != (__int64 *)v23 )
    _libc_free(v15, v14);
}
