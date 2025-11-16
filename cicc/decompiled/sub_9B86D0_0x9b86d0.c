// Function: sub_9B86D0
// Address: 0x9b86d0
//
__int64 __fastcall sub_9B86D0(unsigned int a1, void *a2, unsigned __int64 a3, __int64 a4)
{
  signed __int64 v6; // r12
  unsigned __int64 v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rax

  if ( (_DWORD)a3 == a1 )
  {
    v6 = 4 * a3;
    v7 = *(unsigned int *)(a4 + 12);
    *(_DWORD *)(a4 + 8) = 0;
    v8 = 0;
    LODWORD(v9) = 0;
    if ( v6 >> 2 > v7 )
    {
      sub_C8D5F0(a4, a4 + 16, v6 >> 2, 4);
      v9 = *(unsigned int *)(a4 + 8);
      v8 = 4 * v9;
    }
    if ( v6 )
    {
      memcpy((void *)(*(_QWORD *)a4 + v8), a2, v6);
      LODWORD(v9) = *(_DWORD *)(a4 + 8);
    }
    *(_DWORD *)(a4 + 8) = v9 + (v6 >> 2);
    return 1;
  }
  else if ( (unsigned int)a3 > a1 )
  {
    return sub_9B8470((unsigned int)a3 / a1, (char *)a2, a3, a4);
  }
  else
  {
    sub_9B8300(a1 / (unsigned int)a3, (unsigned int *)a2, a3, a4);
    return 1;
  }
}
