// Function: sub_AC36F0
// Address: 0xac36f0
//
__int64 __fastcall sub_AC36F0(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r13
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // rsi
  _QWORD *i; // rbx
  unsigned __int8 v14; // [rsp+Fh] [rbp-41h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-38h]

  v5 = sub_C33340(a1, a2, a3, a4, a5);
  if ( *a2 == v5 )
    sub_C3C790(&v15, a2);
  else
    sub_C33EB0(&v15, a2);
  v8 = v15;
  switch ( *(_BYTE *)(a1 + 8) )
  {
    case 0:
      v9 = 1;
      v11 = sub_C332F0(&v15, a2, v6, v7);
      if ( v11 != v8 )
        goto LABEL_15;
      break;
    case 1:
      v9 = 1;
      v11 = sub_C33300();
      if ( v11 != v8 )
        goto LABEL_15;
      break;
    case 2:
      v9 = 1;
      v11 = sub_C33310(&v15, a2);
      if ( v11 != v8 )
        goto LABEL_15;
      break;
    case 3:
      if ( v8 == sub_C332F0(&v15, a2, v6, v7) || v8 == sub_C33300() || v8 == sub_C33310(&v15, a2) )
      {
        v9 = 1;
      }
      else
      {
        v9 = 1;
        v11 = sub_C33320(&v15);
        if ( v11 != v8 )
        {
LABEL_15:
          sub_C41640(&v15, v11, 1, &v14);
          v8 = v15;
          v9 = v14 ^ 1;
        }
      }
      break;
    case 4:
      v9 = 1;
      if ( v8 != sub_C332F0(&v15, a2, v6, v7)
        && v8 != sub_C33300()
        && v8 != sub_C33310(&v15, a2)
        && v8 != sub_C33320(&v15) )
      {
        LOBYTE(v9) = v8 == sub_C33420();
      }
      break;
    case 5:
      v9 = 1;
      if ( v8 != sub_C332F0(&v15, a2, v6, v7)
        && v8 != sub_C33300()
        && v8 != sub_C33310(&v15, a2)
        && v8 != sub_C33320(&v15) )
      {
        LOBYTE(v9) = v8 == sub_C33330();
      }
      break;
    case 6:
      v9 = 1;
      if ( v8 != sub_C332F0(&v15, a2, v6, v7) && v8 != sub_C33300() && v8 != sub_C33310(&v15, a2) )
      {
        v10 = sub_C33320(&v15);
        LOBYTE(v9) = v5 == v8;
        LOBYTE(v10) = v8 == v10;
        v9 |= v10;
      }
      break;
    default:
      v9 = 0;
      break;
  }
  if ( v5 != v8 )
  {
    sub_C338F0(&v15);
    return v9;
  }
  if ( !v16 )
    return v9;
  for ( i = &v16[3 * *(v16 - 1)]; v16 != i; sub_91D830(i) )
    i -= 3;
  j_j_j___libc_free_0_0(i - 1);
  return v9;
}
