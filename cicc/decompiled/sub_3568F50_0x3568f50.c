// Function: sub_3568F50
// Address: 0x3568f50
//
void __fastcall sub_3568F50(__int64 *a1)
{
  __int64 v1; // rbp
  __int64 v2; // rsi
  unsigned __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  __int64 v5; // [rsp-48h] [rbp-48h] BYREF
  int v6; // [rsp-40h] [rbp-40h] BYREF
  unsigned __int64 v7; // [rsp-38h] [rbp-38h]
  int *v8; // [rsp-30h] [rbp-30h]
  int *v9; // [rsp-28h] [rbp-28h]
  __int64 v10; // [rsp-20h] [rbp-20h]
  __int64 v11; // [rsp-8h] [rbp-8h]

  if ( unk_503EEF4 )
  {
    v11 = v1;
    v2 = *a1;
    v6 = 0;
    v7 = 0;
    v8 = &v6;
    v9 = &v6;
    v10 = 0;
    sub_3568E80(a1, v2 & 0xFFFFFFFFFFFFFFF8LL, &v5);
    v3 = v7;
    while ( v3 )
    {
      sub_35672F0(*(_QWORD *)(v3 + 24));
      v4 = v3;
      v3 = *(_QWORD *)(v3 + 16);
      j_j___libc_free_0(v4);
    }
  }
}
