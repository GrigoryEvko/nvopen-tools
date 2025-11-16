// Function: sub_AC5470
// Address: 0xac5470
//
__int64 __fastcall sub_AC5470(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // rsi
  __int64 v6; // rdi
  unsigned int *v7; // rbx
  unsigned __int8 v8; // al
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+0h] [rbp-30h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-28h]

  v4 = a3;
  v6 = a2;
  v7 = (unsigned int *)sub_AC5300(a2, a3);
  v8 = *(_BYTE *)(sub_AC5230(a2) + 8);
  if ( v8 == 2 )
  {
    v19 = *v7;
    v22 = 32;
    v21 = v19;
    v12 = sub_C33310(a2, v4);
  }
  else if ( v8 > 2u )
  {
    if ( v8 != 3 )
      BUG();
    v18 = *(_QWORD *)v7;
    v22 = 64;
    v21 = v18;
    v12 = sub_C33320(a2);
  }
  else if ( v8 )
  {
    v11 = *(unsigned __int16 *)v7;
    v22 = 16;
    v21 = v11;
    v12 = sub_C33300();
  }
  else
  {
    v20 = *(unsigned __int16 *)v7;
    v22 = 16;
    v21 = v20;
    v12 = sub_C332F0(a2, v4, v9, v10);
  }
  v16 = v12;
  if ( v12 == sub_C33340(v6, v4, v13, v14, v15) )
    sub_C3C640(a1, v16, &v21);
  else
    sub_C3B160(a1, v16, &v21);
  if ( v22 > 0x40 && v21 )
    j_j___libc_free_0_0(v21);
  return a1;
}
