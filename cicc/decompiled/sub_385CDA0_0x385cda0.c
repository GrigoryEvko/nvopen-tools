// Function: sub_385CDA0
// Address: 0x385cda0
//
__int64 __fastcall sub_385CDA0(unsigned int *src, char *a2, char *a3, unsigned int *a4, _QWORD *a5)
{
  __int64 v8; // rax
  unsigned int *v9; // r10
  __int64 v10; // r9
  __int64 v12; // [rsp-10h] [rbp-50h]
  __int64 v13; // [rsp+0h] [rbp-40h]
  unsigned int *srca; // [rsp+8h] [rbp-38h]

  v8 = (((a2 - (char *)src) >> 2) + 1) / 2;
  v13 = 4 * v8;
  srca = &src[v8];
  if ( v8 <= (__int64)a4 )
  {
    sub_385BAB0(src, (char *)&src[v8], a3, a5);
    sub_385BAB0(srca, a2, a3, a5);
    v10 = v13;
    v9 = srca;
  }
  else
  {
    sub_385CDA0(src);
    sub_385CDA0(srca);
    v9 = srca;
    v10 = v13;
  }
  sub_385C920((char *)src, v9, (__int64)a2, v10 >> 2, (a2 - (char *)v9) >> 2, (unsigned int *)a3, a4, a5);
  return v12;
}
