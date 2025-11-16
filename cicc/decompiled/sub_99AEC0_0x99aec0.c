// Function: sub_99AEC0
// Address: 0x99aec0
//
unsigned __int64 __fastcall sub_99AEC0(_BYTE *a1, __int64 *a2, __int64 *a3, unsigned int *a4, unsigned int a5)
{
  unsigned int *v9; // rax
  unsigned int v10; // eax
  _BYTE *v12; // rdi
  unsigned __int64 v13; // rax
  unsigned int v14; // ecx
  unsigned __int64 v15; // rax

  v9 = (unsigned int *)sub_C94E20(qword_4F862D0);
  if ( v9 )
    v10 = *v9;
  else
    v10 = qword_4F862D0[2];
  if ( a5 >= v10 || *a1 != 86 )
    return 0;
  v12 = (_BYTE *)*((_QWORD *)a1 - 12);
  if ( (unsigned __int8)(*v12 - 82) <= 1u )
  {
    v15 = sub_99AC20((__int64)v12, *((_QWORD *)a1 - 8), *((unsigned __int8 **)a1 - 4), a2, a3, a4);
    v14 = v15;
    v13 = HIDWORD(v15);
  }
  else
  {
    LODWORD(v13) = 0;
    v14 = 0;
  }
  return __PAIR64__(v13, v14);
}
