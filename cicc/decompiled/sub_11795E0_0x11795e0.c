// Function: sub_11795E0
// Address: 0x11795e0
//
_QWORD *__fastcall sub_11795E0(__int64 a1, unsigned int a2, char a3, char a4)
{
  __int16 v4; // r12
  __int64 v5; // r13
  __int64 v6; // r14
  __int64 *v7; // rcx
  __int64 v8; // rdx
  __int64 *v9; // rdx
  __int64 v10; // r14
  _QWORD *v11; // rbx
  _QWORD **v12; // rdx
  int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // rsi
  unsigned int **v17; // [rsp+8h] [rbp-78h]
  __int64 v18; // [rsp+18h] [rbp-68h]
  _QWORD v19[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v20; // [rsp+40h] [rbp-40h]

  if ( !a3 )
    return 0;
  v4 = a2;
  v5 = sub_1179360(*(_QWORD *)a1, **(_QWORD **)(a1 + 8), **(_QWORD **)(a1 + 16), a2 - 32 <= 1, a4 & (a2 - 32 > 1));
  if ( !v5 )
    return 0;
  v6 = *(_QWORD *)(a1 + 56);
  v17 = *(unsigned int ***)(*(_QWORD *)(a1 + 24) + 32LL);
  v19[0] = sub_BD5D20(v6);
  v20 = 773;
  v7 = *(__int64 **)(a1 + 48);
  v19[1] = v8;
  v9 = *(__int64 **)(a1 + 40);
  v19[2] = ".v";
  v10 = sub_B36550(v17, **(_QWORD **)(a1 + 32), *v9, *v7, (__int64)v19, v6);
  if ( !**(_BYTE **)(a1 + 64) )
    v4 = sub_B52F50(a2);
  v20 = 257;
  v11 = sub_BD2C40(72, unk_3F10FD0);
  if ( v11 )
  {
    v12 = *(_QWORD ***)(v5 + 8);
    v13 = *((unsigned __int8 *)v12 + 8);
    if ( (unsigned int)(v13 - 17) > 1 )
    {
      v15 = sub_BCB2A0(*v12);
    }
    else
    {
      BYTE4(v18) = (_BYTE)v13 == 18;
      LODWORD(v18) = *((_DWORD *)v12 + 8);
      v14 = (__int64 *)sub_BCB2A0(*v12);
      v15 = sub_BCE1B0(v14, v18);
    }
    sub_B523C0((__int64)v11, v15, 53, v4, v5, v10, (__int64)v19, 0, 0, 0);
  }
  return v11;
}
