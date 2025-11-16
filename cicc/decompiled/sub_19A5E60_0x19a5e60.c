// Function: sub_19A5E60
// Address: 0x19a5e60
//
_QWORD *__fastcall sub_19A5E60(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  _BOOL4 v5; // r8d
  __int64 v6; // r13
  _BOOL4 v8; // [rsp+Ch] [rbp-34h]

  v2 = sub_19A5DC0(a1, a2);
  if ( !v3 )
    return v2;
  v4 = v3;
  v5 = 1;
  if ( !v2 && v3 != a1 + 8 )
    v5 = *a2 < *(_QWORD *)(v3 + 32);
  v8 = v5;
  v6 = sub_22077B0(40);
  *(_QWORD *)(v6 + 32) = *a2;
  sub_220F040(v8, v6, v4, a1 + 8);
  ++*(_QWORD *)(a1 + 40);
  return (_QWORD *)v6;
}
