// Function: sub_2C12A50
// Address: 0x2c12a50
//
__int64 __fastcall sub_2C12A50(__int64 a1, __int64 a2, __int64 **a3)
{
  __int64 *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdi
  char v8; // dl
  __int64 v9; // rax
  __int64 v10; // rbx
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // r14
  int v14; // eax
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // r15
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v21; // r12
  __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdx
  _BYTE *v25; // rax
  __int64 v26; // [rsp+0h] [rbp-90h]
  __int64 v27; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+20h] [rbp-70h] BYREF
  __int64 v30; // [rsp+28h] [rbp-68h]
  unsigned __int64 v31; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v32; // [rsp+38h] [rbp-58h]
  __int64 *v33; // [rsp+40h] [rbp-50h]
  unsigned __int64 v34; // [rsp+48h] [rbp-48h] BYREF
  unsigned int v35; // [rsp+50h] [rbp-40h]

  v4 = *(__int64 **)(a1 + 48);
  v5 = *v4;
  v30 = 0;
  v29 = v5;
  v6 = sub_2BF04A0(*v4);
  v7 = v29;
  if ( v6 )
  {
    v8 = *(_BYTE *)(v6 + 8);
    v7 = v29;
    switch ( v8 )
    {
      case 9:
        if ( **(_BYTE **)(v6 + 136) != 86 )
          break;
        goto LABEL_19;
      case 4:
        if ( *(_BYTE *)(v6 + 160) != 57 )
          break;
        goto LABEL_19;
      case 24:
LABEL_19:
        v7 = *(_QWORD *)(*(_QWORD *)(sub_2BF0490(v29) + 48) + 8LL);
        v29 = v7;
        break;
    }
  }
  v35 = 64;
  v33 = &v29;
  v34 = 0;
  v9 = sub_2BF04A0(v7);
  v10 = v9;
  if ( !v9 )
    goto LABEL_7;
  v11 = *(_BYTE *)(v9 + 8);
  if ( v11 == 23 )
  {
LABEL_6:
    if ( *(_DWORD *)(v10 + 160) != 15 )
      goto LABEL_7;
    goto LABEL_22;
  }
  if ( v11 != 9 )
  {
    if ( v11 != 16 )
    {
      if ( v11 != 4 || *(_BYTE *)(v10 + 160) != 15 )
        goto LABEL_7;
      goto LABEL_22;
    }
    goto LABEL_6;
  }
  if ( **(_BYTE **)(v10 + 136) != 44 )
    goto LABEL_7;
LABEL_22:
  v32 = v35;
  if ( v35 > 0x40 )
    sub_C43780((__int64)&v31, (const void **)&v34);
  else
    v31 = v34;
  v21 = **(_QWORD **)(v10 + 48);
  if ( !sub_2BF04A0(v21) )
  {
    v22 = *(_QWORD *)(v21 + 40);
    if ( v22 )
    {
      if ( *(_BYTE *)v22 == 17
        || (v24 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v22 + 8) + 8LL) - 17, (unsigned int)v24 <= 1)
        && *(_BYTE *)v22 <= 0x15u
        && (v25 = sub_AD7630(v22, 0, v24), (v22 = (__int64)v25) != 0)
        && *v25 == 17 )
      {
        if ( sub_1112D90(v22 + 24, (__int64)&v31) )
        {
          v23 = *(_QWORD *)(*(_QWORD *)(v10 + 48) + 8LL);
          if ( v23 )
            *v33 = v23;
        }
      }
    }
  }
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
LABEL_7:
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  v12 = sub_2BF0490(v29);
  v13 = v12;
  if ( *(_BYTE *)(v12 + 8) == 23 )
  {
    v14 = *(_DWORD *)(v12 + 160);
    BYTE4(v30) = 1;
    LODWORD(v30) = v14;
  }
  v15 = sub_2BF0490(**(_QWORD **)(v13 + 48));
  v16 = sub_2BF0490(*(_QWORD *)(*(_QWORD *)(v13 + 48) + 8LL));
  v17 = (__int64)(a3 + 2);
  v27 = sub_2BFD6A0((__int64)(a3 + 2), *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL));
  if ( v15 )
    v18 = *(__int64 **)(v15 + 48);
  else
    v18 = *(__int64 **)(v13 + 48);
  v26 = sub_2BFD6A0(v17, *v18);
  if ( v16 )
    v19 = sub_2BFD6A0(v17, **(_QWORD **)(v16 + 48));
  else
    v19 = sub_2BFD6A0(v17, *(_QWORD *)(*(_QWORD *)(v13 + 48) + 8LL));
  return sub_DFB6F0(*a3, *(unsigned int *)(a1 + 152), v26, v19, v27, a2);
}
