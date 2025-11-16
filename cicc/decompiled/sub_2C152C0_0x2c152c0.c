// Function: sub_2C152C0
// Address: 0x2c152c0
//
__int64 __fastcall sub_2C152C0(__int64 a1, __int64 a2, __int64 **a3)
{
  int v4; // edx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rdx
  __int64 v11; // rcx
  _QWORD *v12; // rdi
  _QWORD *v13; // rsi
  _QWORD *v14; // rax

  if ( !*(_QWORD *)(a1 + 136) )
    return 0;
  v4 = *(_DWORD *)(a1 + 160);
  v6 = **(_QWORD **)(a1 + 48);
  if ( v4 != 38 && v4 != 45 )
    goto LABEL_5;
  v11 = *(unsigned int *)(a1 + 120);
  if ( !(_DWORD)v11 )
    goto LABEL_5;
  v12 = *(_QWORD **)(a1 + 112);
  v13 = &v12[v11];
  v14 = v12 + 1;
  if ( v12 + 1 != v13 )
  {
    while ( *v12 == *v14 )
    {
      if ( ++v14 == v13 )
        goto LABEL_6;
    }
    if ( v14 != v13 )
    {
LABEL_5:
      if ( (unsigned int)(v4 - 39) <= 1 || v4 == 46 )
      {
        if ( sub_2BF04A0(v6) )
        {
          if ( sub_2BF0490(v6) )
            sub_2BF0490(v6);
        }
      }
    }
  }
LABEL_6:
  v7 = sub_2BFD6A0((__int64)(a3 + 2), v6);
  v8 = sub_2AAEDF0(v7, a2);
  v9 = sub_2AAEDF0(*(_QWORD *)(a1 + 168), a2);
  return sub_DFD060(*a3, *(unsigned int *)(a1 + 160), v9, v8);
}
