// Function: sub_14A9CE0
// Address: 0x14a9ce0
//
void __fastcall sub_14A9CE0(__int64 a1)
{
  unsigned int v1; // ecx
  unsigned __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rcx

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
  {
    *(_QWORD *)a1 = -1;
    v2 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v1;
LABEL_3:
    *(_QWORD *)a1 &= v2;
    return;
  }
  memset(*(void **)a1, -1, 8 * (((unsigned __int64)v1 + 63) >> 6));
  v3 = *(unsigned int *)(a1 + 8);
  v2 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a1 + 8);
  if ( (unsigned int)v3 <= 0x40 )
    goto LABEL_3;
  v4 = (unsigned int)((unsigned __int64)(v3 + 63) >> 6) - 1;
  *(_QWORD *)(*(_QWORD *)a1 + 8 * v4) &= v2;
}
