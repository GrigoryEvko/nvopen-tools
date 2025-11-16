// Function: sub_26F1910
// Address: 0x26f1910
//
unsigned __int64 __fastcall sub_26F1910(__int64 a1, const void *a2, unsigned __int64 a3, const void *a4, size_t a5)
{
  unsigned __int64 result; // rax
  __int64 v10; // rax
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // r8
  unsigned int *v16; // r9
  __int64 v17; // rbx
  unsigned int *v18; // [rsp+8h] [rbp-48h]
  __int64 v19[8]; // [rsp+10h] [rbp-40h] BYREF

  result = (unsigned __int64)sub_BA8CB0(*(_QWORD *)a1, (__int64)a2, a3);
  if ( result && *(_QWORD *)(result + 16) )
  {
    v18 = *(unsigned int **)(a1 + 8);
    v10 = sub_B9B140(*(__int64 **)(a1 + 16), a2, a3);
    v11 = *(__int64 **)(a1 + 16);
    v19[0] = v10;
    v12 = sub_B9B140(v11, a4, a5);
    v13 = *(__int64 **)(a1 + 16);
    v19[1] = v12;
    v14 = sub_B9C770(v13, v19, (__int64 *)2, 0, 1);
    v16 = v18;
    v17 = v14;
    result = v18[2];
    if ( result + 1 > v18[3] )
    {
      sub_C8D5F0((__int64)v18, v18 + 4, result + 1, 8u, v15, (__int64)v18);
      v16 = v18;
      result = v18[2];
    }
    *(_QWORD *)(*(_QWORD *)v16 + 8 * result) = v17;
    ++v16[2];
  }
  return result;
}
