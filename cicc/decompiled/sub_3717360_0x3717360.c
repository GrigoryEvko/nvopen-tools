// Function: sub_3717360
// Address: 0x3717360
//
bool __fastcall sub_3717360(_DWORD *a1, int a2)
{
  unsigned int v2; // ebx
  unsigned int (*v3)(void); // rax

  v2 = (a2 & 0x7FFFFFFF) - 4096;
  v3 = *(unsigned int (**)(void))(*(_QWORD *)a1 + 56LL);
  if ( (char *)v3 == (char *)sub_3717150 )
    return a1[36] >= v2;
  else
    return v3() >= v2;
}
