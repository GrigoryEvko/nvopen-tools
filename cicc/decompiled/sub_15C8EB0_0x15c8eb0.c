// Function: sub_15C8EB0
// Address: 0x15c8eb0
//
__int64 __fastcall sub_15C8EB0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rdx
  int v3; // eax
  char v5; // [rsp+Fh] [rbp-41h] BYREF
  int v6; // [rsp+10h] [rbp-40h] BYREF
  int v7; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v8; // [rsp+18h] [rbp-38h] BYREF
  _QWORD v9[6]; // [rsp+20h] [rbp-30h] BYREF

  (*(void (__fastcall **)(__int64 *))(*a1 + 144))(a1);
  v2 = *a2;
  v9[1] = a2[1];
  v3 = *((_DWORD *)a2 + 4);
  v9[0] = v2;
  v6 = v3;
  v7 = *((_DWORD *)a2 + 5);
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __int64 *))(*a1 + 120))(
         a1,
         "File",
         1,
         0,
         &v5,
         &v8) )
  {
    sub_15C8D20(a1, (__int64)v9);
    (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v8);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __int64 *))(*a1 + 120))(
         a1,
         "Line",
         1,
         0,
         &v5,
         &v8) )
  {
    sub_15C8760(a1, (__int64)&v6);
    (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v8);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __int64 *))(*a1 + 120))(
         a1,
         "Column",
         1,
         0,
         &v5,
         &v8) )
  {
    sub_15C8760(a1, (__int64)&v7);
    (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v8);
  }
  return (*(__int64 (__fastcall **)(__int64 *))(*a1 + 152))(a1);
}
