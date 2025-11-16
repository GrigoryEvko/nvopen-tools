// Function: sub_37391A0
// Address: 0x37391a0
//
__int64 __fastcall sub_37391A0(__int64 *a1, __int64 a2, unsigned __int8 *a3, char a4, __int64 a5, __int64 a6, int a7)
{
  __int16 v9; // ax
  __int64 v10; // r12
  __int16 v11; // ax
  __int16 v12; // ax
  unsigned __int8 *v14; // r15
  __int16 v15; // ax
  __int16 v16; // ax
  unsigned __int8 v17; // al
  __int64 v18; // rbx
  _BYTE *v19; // rdx
  __int64 v20; // rax
  size_t v21; // rdx
  size_t v22; // rcx
  _DWORD v25[16]; // [rsp+10h] [rbp-40h] BYREF

  v9 = sub_3736180((__int64)a1, 0x48u);
  v10 = sub_324C6D0(a1, v9, a2, 0);
  if ( !a7 )
  {
    v14 = sub_3250680(a1, a3, 0);
    if ( *(_DWORD *)(a1[26] + 6224) == 3 )
    {
      if ( (_DWORD)qword_5050DA8 == 2 )
        goto LABEL_8;
    }
    else if ( (_DWORD)qword_5050DA8 != 1 )
    {
      goto LABEL_8;
    }
    if ( (a3[36] & 8) == 0 )
    {
      sub_3215160((__int64)v25, (__int64)v14, 110);
      if ( !v25[0] )
      {
        v17 = *(a3 - 16);
        if ( (v17 & 2) != 0 )
          v18 = *((_QWORD *)a3 - 4);
        else
          v18 = (__int64)&a3[-8 * ((v17 >> 2) & 0xF) - 16];
        v19 = *(_BYTE **)(v18 + 24);
        if ( v19 )
        {
          v20 = sub_B91420(*(_QWORD *)(v18 + 24));
          v22 = v21;
          v19 = (_BYTE *)v20;
        }
        else
        {
          v22 = 0;
        }
        sub_324B070(a1, (__int64)v14, v19, v22);
      }
    }
LABEL_8:
    v15 = sub_37361D0((__int64)a1, 0x7Fu);
    sub_32494F0(a1, v10, v15, (unsigned __int64)v14);
    if ( !a4 )
      goto LABEL_3;
    goto LABEL_9;
  }
  LOBYTE(v25[0]) = 1;
  v25[1] = a7;
  v11 = sub_37361D0((__int64)a1, 0x83u);
  sub_3738310(a1, v10, v11, (__int64)v25);
  if ( !a4 )
  {
LABEL_3:
    v12 = sub_37361D0((__int64)a1, 0x7Du);
    sub_3738C10(a1, v10, v12, a5);
    return v10;
  }
LABEL_9:
  v16 = sub_37361D0((__int64)a1, 0x82u);
  sub_3249FA0(a1, v10, v16);
  if ( !sub_3736140((__int64)a1) )
    sub_3738C10(a1, v10, 129, a6);
  if ( sub_3736140((__int64)a1) )
    goto LABEL_3;
  return v10;
}
