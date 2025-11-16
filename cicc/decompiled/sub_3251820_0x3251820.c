// Function: sub_3251820
// Address: 0x3251820
//
__int64 __fastcall sub_3251820(__int64 *a1, unsigned __int8 *a2, __int64 a3, unsigned __int8 *a4)
{
  __int16 v6; // ax
  __int64 v7; // r13
  unsigned __int8 v8; // al
  unsigned __int8 v9; // al
  unsigned __int8 *v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // r15
  size_t v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int16 v17; // bx
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int *v22; // [rsp+0h] [rbp-40h]
  size_t v23; // [rsp+8h] [rbp-38h]

  if ( a3 )
  {
    v6 = sub_AF18C0((__int64)a4);
    v7 = sub_324C6D0(a1, v6, a3, a4);
  }
  else
  {
    v17 = sub_AF18C0((__int64)a4);
    v18 = sub_A777F0(0x30u, a1 + 11);
    v7 = v18;
    if ( v18 )
    {
      *(_QWORD *)(v18 + 8) = 0;
      *(_QWORD *)v18 = v18 | 4;
      *(_QWORD *)(v18 + 16) = 0;
      *(_DWORD *)(v18 + 24) = -1;
      *(_WORD *)(v18 + 28) = v17;
      *(_BYTE *)(v18 + 30) = 0;
      *(_QWORD *)(v18 + 32) = 0;
      *(_QWORD *)(v18 + 40) = 0;
    }
    sub_324C3F0((__int64)a1, a4, v18);
    sub_3251400((__int64)a1, (__int64)a2, v7, v19, v20, v21);
  }
  v8 = *a4;
  if ( *a4 == 14 )
  {
    if ( !*(_BYTE *)(a1[26] + 3691) || (a4[20] & 4) != 0 )
      goto LABEL_12;
    v9 = *(a4 - 16);
    if ( (v9 & 2) != 0 )
    {
      v10 = (unsigned __int8 *)*((_QWORD *)a4 - 4);
      if ( *((_QWORD *)v10 + 2) )
        goto LABEL_8;
    }
    else
    {
      v15 = 8LL * ((v9 >> 2) & 0xF);
      v10 = &a4[-v15 - 16];
      if ( *(_QWORD *)&a4[-v15] )
        goto LABEL_8;
    }
    if ( !*((_QWORD *)v10 + 7) )
    {
LABEL_12:
      sub_3248280((__int64)a1, a2, (__int64)a4, v7);
      sub_32507E0(a1, v7, (__int64)a4);
      return v7;
    }
LABEL_8:
    v11 = *((_QWORD *)v10 + 7);
    if ( v11 )
    {
      sub_3248250((__int64)a1, (__int64)a4, v7, a2);
      v12 = a1[26];
      v22 = (int *)sub_B91420(v11);
      v23 = v13;
      v14 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
      sub_3236930(v12, v14, v22, v23, v7, (__int64)a4);
    }
    else
    {
      sub_3248280((__int64)a1, a2, (__int64)a4, v7);
      (*(void (__fastcall **)(__int64 *, __int64, unsigned __int8 *))(*a1 + 88))(a1, v7, a4);
    }
    return v7;
  }
  switch ( v8 )
  {
    case 0xCu:
      sub_3248280((__int64)a1, a2, (__int64)a4, v7);
      sub_324B0E0(a1, v7, (__int64)a4);
      break;
    case 0x22u:
      sub_3248280((__int64)a1, a2, (__int64)a4, v7);
      sub_324B550(a1, v7, (__int64)a4);
      break;
    case 0xFu:
      sub_3248280((__int64)a1, a2, (__int64)a4, v7);
      sub_324C9F0(a1, v7, (__int64)a4);
      break;
    case 0x24u:
      sub_324B240(a1, v7, (__int64)a4, 0);
      break;
    default:
      sub_3248280((__int64)a1, a2, (__int64)a4, v7);
      sub_324D800(a1, v7, (__int64)a4);
      break;
  }
  return v7;
}
