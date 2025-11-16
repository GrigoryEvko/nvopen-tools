// Function: sub_35D3520
// Address: 0x35d3520
//
__int64 __fastcall sub_35D3520(__int64 a1, __int64 *a2)
{
  char *v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = (char *)sub_2E791E0(a2);
  if ( !sub_BC63A0(v2, v3) )
    return 0;
  v5 = sub_B2BE50(*a2);
  v6 = sub_B6F970(v5);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v6 + 24LL))(
          v6,
          "stack-frame-layout",
          18) )
    return 0;
  sub_35D1120(a1, (__int64)a2);
  return 0;
}
