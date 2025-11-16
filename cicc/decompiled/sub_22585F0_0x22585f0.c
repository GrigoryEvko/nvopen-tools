// Function: sub_22585F0
// Address: 0x22585f0
//
void *__fastcall sub_22585F0(__int64 a1, char *a2)
{
  size_t v2; // rax
  char *v3; // rdi
  size_t v4; // r13
  __int64 v6[2]; // [rsp+0h] [rbp-60h] BYREF
  char *v7; // [rsp+10h] [rbp-50h]
  __int64 v8; // [rsp+18h] [rbp-48h]
  void *dest; // [rsp+20h] [rbp-40h]
  __int64 v10; // [rsp+28h] [rbp-38h]
  __int64 v11; // [rsp+30h] [rbp-30h]

  v10 = 0x100000000LL;
  v11 = a1;
  v6[1] = 0;
  v7 = 0;
  v8 = 0;
  dest = 0;
  v6[0] = (__int64)&unk_49DD210;
  sub_CB5980((__int64)v6, 0, 0, 0);
  if ( !a2 )
    goto LABEL_6;
  v2 = strlen(a2);
  v3 = (char *)dest;
  v4 = v2;
  if ( v2 > v8 - (__int64)dest )
  {
    sub_CB6200((__int64)v6, (unsigned __int8 *)a2, v2);
LABEL_6:
    v3 = (char *)dest;
    goto LABEL_7;
  }
  if ( v2 )
  {
    memcpy(dest, a2, v2);
    v3 = (char *)dest + v4;
    dest = (char *)dest + v4;
  }
LABEL_7:
  if ( v7 != v3 )
    sub_CB5AE0(v6);
  v6[0] = (__int64)&unk_49DD210;
  return sub_CB5840((__int64)v6);
}
