// Function: sub_2F5A580
// Address: 0x2f5a580
//
void __fastcall sub_2F5A580(_QWORD *a1)
{
  __int64 **v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // r13

  v1 = (__int64 **)a1[102];
  v2 = sub_B2BE50(**v1);
  if ( sub_B6EA50(v2)
    || (v3 = sub_B2BE50(**v1),
        v4 = sub_B6F970(v3),
        (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v4 + 32LL))(v4, "regalloc", 8))
    || (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v4 + 40LL))(v4, "regalloc", 8)
    || (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64))(*(_QWORD *)v4 + 24LL))(v4, "regalloc", 8) )
  {
    sub_2F59FD0(a1);
  }
}
