// Function: sub_108ACE0
// Address: 0x108ace0
//
__int64 *__fastcall sub_108ACE0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a2;
  *a2 = 0;
  v8[0] = v4;
  v5 = sub_22077B0(136);
  v6 = v5;
  if ( v5 )
    sub_1085F30(v5, v8, a3);
  if ( v8[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8[0] + 8LL))(v8[0]);
  *a1 = v6;
  return a1;
}
