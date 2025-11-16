// Function: sub_ED5BE0
// Address: 0xed5be0
//
unsigned __int64 *__fastcall sub_ED5BE0(unsigned __int64 *a1, __int64 a2, __int64 a3, void *a4, size_t a5)
{
  void *v7; // rax
  size_t v8; // rdx
  void *v9; // r8
  size_t v10; // r15
  int v12; // eax
  void *v13; // [rsp+8h] [rbp-58h]
  unsigned __int64 v14; // [rsp+18h] [rbp-48h] BYREF
  __int64 v15[8]; // [rsp+20h] [rbp-40h] BYREF

  v15[0] = a2;
  v15[1] = a3;
  sub_ED5910(&v14, v15, a4, a5);
  if ( (v14 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v14 & 0xFFFFFFFFFFFFFFFELL | 1;
  }
  else
  {
    v7 = (void *)sub_ED18E0((__int64)a4, a5);
    v9 = v7;
    v10 = v8;
    if ( v8 != a5 || v8 && (v13 = v7, v12 = memcmp(v7, a4, v8), v9 = v13, v12) )
      sub_ED5910(a1, v15, v9, v10);
    else
      *a1 = 1;
  }
  return a1;
}
