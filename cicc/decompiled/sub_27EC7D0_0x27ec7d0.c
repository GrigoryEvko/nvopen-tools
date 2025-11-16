// Function: sub_27EC7D0
// Address: 0x27ec7d0
//
char __fastcall sub_27EC7D0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rcx
  _QWORD *v14; // rax
  int v15; // esi
  unsigned int v16; // edx
  _QWORD *v17; // rdi
  __int64 *v18; // rsi

  sub_31032E0(a4);
  if ( !a2 )
    BUG();
  sub_31032A0(a4, a1, *(_QWORD *)(a2 + 16));
  sub_B44550(a1, *(_QWORD *)(a2 + 16), (unsigned __int64 *)a2, a3);
  v13 = *(_QWORD *)(*a5 + 40);
  LODWORD(v14) = *(_DWORD *)(*a5 + 56);
  if ( (_DWORD)v14 )
  {
    v15 = (_DWORD)v14 - 1;
    v16 = ((_DWORD)v14 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v14 = (_QWORD *)(v13 + 16LL * v16);
    v17 = (_QWORD *)*v14;
    if ( a1 == (_QWORD *)*v14 )
    {
LABEL_4:
      v18 = (__int64 *)v14[1];
      if ( v18 )
        LOBYTE(v14) = (unsigned __int8)sub_D75590(a5, v18, *(_QWORD *)(a2 + 16), 2, v11, v12);
    }
    else
    {
      LODWORD(v14) = 1;
      while ( v17 != (_QWORD *)-4096LL )
      {
        v11 = (unsigned int)((_DWORD)v14 + 1);
        v16 = v15 & ((_DWORD)v14 + v16);
        v14 = (_QWORD *)(v13 + 16LL * v16);
        v17 = (_QWORD *)*v14;
        if ( a1 == (_QWORD *)*v14 )
          goto LABEL_4;
        LODWORD(v14) = v11;
      }
    }
  }
  if ( a6 )
    LOBYTE(v14) = sub_D9D700(a6, (__int64)a1);
  return (char)v14;
}
