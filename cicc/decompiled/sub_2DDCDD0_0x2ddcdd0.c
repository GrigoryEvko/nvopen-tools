// Function: sub_2DDCDD0
// Address: 0x2ddcdd0
//
void __fastcall sub_2DDCDD0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rdx
  unsigned __int64 v4; // rdi
  __int64 v5; // r13
  const void *v6; // r15
  size_t v7; // r14
  __int64 v8; // r12
  int v9; // eax
  int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rdi

  v3 = (_QWORD *)sub_22077B0(0x70u);
  if ( v3 )
  {
    memset(v3, 0, 0x70u);
    v3[4] = v3 + 6;
    v3[5] = 0x100000000LL;
    v3[12] = 0x1000000000LL;
  }
  v4 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 8) = v3;
  if ( v4 )
    sub_2DDBD80(v4);
  if ( !(_BYTE)qword_501E1C8 )
  {
    v5 = *(_QWORD *)(a1 + 16);
    if ( !v5
      || ((v6 = *(const void **)(a2 + 168),
           v7 = *(_QWORD *)(a2 + 176),
           v8 = *(_QWORD *)(v5 + 48) + 8LL * *(unsigned int *)(v5 + 56),
           v9 = sub_C92610(),
           v10 = sub_C92860((__int64 *)(v5 + 48), v6, v7, v9),
           v10 == -1)
        ? (v11 = *(_QWORD *)(v5 + 48) + 8LL * *(unsigned int *)(v5 + 56))
        : (v11 = *(_QWORD *)(v5 + 48) + 8LL * v10),
          v8 != v11) )
    {
      if ( *(_BYTE *)(sub_3111D40() + 16) )
      {
        *(_DWORD *)a1 = 1;
      }
      else
      {
        v12 = *(_QWORD *)(sub_3111D40() + 8);
        if ( v12 )
        {
          if ( sub_311A9A0(v12, 0) )
            *(_DWORD *)a1 = 2;
        }
      }
    }
  }
}
