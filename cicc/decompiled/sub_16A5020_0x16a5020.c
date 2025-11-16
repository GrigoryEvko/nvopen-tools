// Function: sub_16A5020
// Address: 0x16a5020
//
unsigned __int64 __fastcall sub_16A5020(__int64 a1, _QWORD *a2, unsigned int a3)
{
  unsigned int v4; // ecx
  unsigned __int64 result; // rax
  size_t v7; // r12
  void *v8; // rax
  void *v9; // rax
  __int64 v10; // r12
  int v11; // edx
  unsigned int v12; // r15d
  unsigned __int64 v13; // r12
  _QWORD *v14; // rdi

  v4 = *(_DWORD *)(a1 + 8);
  if ( v4 <= 0x40 )
  {
    *(_QWORD *)a1 = *a2;
    result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v4;
LABEL_3:
    *(_QWORD *)a1 &= result;
    return result;
  }
  v7 = 8 * (((unsigned __int64)v4 + 63) >> 6);
  v8 = (void *)sub_2207820(v7);
  v9 = memset(v8, 0, v7);
  v10 = *(unsigned int *)(a1 + 8);
  v11 = a3;
  *(_QWORD *)a1 = v9;
  v12 = v10;
  v13 = (unsigned __int64)(v10 + 63) >> 6;
  if ( a3 > (unsigned int)v13 )
    v11 = v13;
  v14 = memcpy(v9, a2, (unsigned int)(8 * v11));
  result = 0xFFFFFFFFFFFFFFFFLL >> -(char)v12;
  if ( v12 <= 0x40 )
    goto LABEL_3;
  v14[(unsigned int)(v13 - 1)] &= result;
  return result;
}
