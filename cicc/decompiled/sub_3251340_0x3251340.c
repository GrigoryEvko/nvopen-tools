// Function: sub_3251340
// Address: 0x3251340
//
unsigned __int64 __fastcall sub_3251340(__int64 *a1, unsigned __int8 *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rdx
  unsigned __int8 *v4; // rbx
  __int64 v5; // r15
  unsigned __int64 v6; // r14
  __int16 v8; // ax

  v2 = *(a2 - 16);
  if ( (v2 & 2) != 0 )
    v3 = *((_QWORD *)a2 - 4);
  else
    v3 = (__int64)&a2[-8 * ((v2 >> 2) & 0xF) - 16];
  v4 = *(unsigned __int8 **)(v3 + 8);
  v5 = (*(__int64 (__fastcall **)(__int64 *, unsigned __int8 *))(*a1 + 48))(a1, v4);
  v6 = (unsigned __int64)sub_3247C80((__int64)a1, a2);
  if ( !v6 )
  {
    v8 = sub_AF18C0((__int64)a2);
    v6 = sub_324C6D0(a1, v8, v5, a2);
    sub_32507E0(a1, v6, (__int64)a2);
    sub_3248280((__int64)a1, v4, (__int64)a2, v6);
  }
  return v6;
}
