// Function: sub_22CFC60
// Address: 0x22cfc60
//
__int64 __fastcall sub_22CFC60(__int64 *a1, unsigned int a2, _BYTE *a3, _BYTE *a4, __int64 a5, char a6)
{
  char v6; // r14
  __int64 v7; // r13
  __int64 v8; // r12
  unsigned int v10; // eax
  __int64 v11; // rbx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  _QWORD **v14; // rdx
  int v15; // ecx
  __int64 *v16; // rax
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-B8h]
  __int64 v19; // [rsp+10h] [rbp-B0h]
  __int64 v20; // [rsp+18h] [rbp-A8h]
  __int64 v21; // [rsp+28h] [rbp-98h]
  unsigned __int8 v22[48]; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int8 v23[96]; // [rsp+60h] [rbp-60h] BYREF

  v6 = a6;
  v7 = (__int64)a3;
  v8 = (__int64)a4;
  if ( *a4 <= 0x15u )
    return sub_22CF7C0(a1, a2, (__int64)a3, (__int64)a4, a5, a6);
  if ( *a3 <= 0x15u )
  {
    v20 = a5;
    v10 = sub_B52F50(a2);
    a6 = v6;
    a4 = (_BYTE *)v7;
    a3 = (_BYTE *)v8;
    a5 = v20;
    a2 = v10;
    return sub_22CF7C0(a1, a2, (__int64)a3, (__int64)a4, a5, a6);
  }
  v11 = 0;
  if ( a6 )
  {
    v18 = a5;
    v19 = sub_B43CA0(a5);
    v12 = sub_22C1480(a1, v19);
    sub_22CDEF0((__int64)v22, v12, v7, *(_QWORD *)(v18 + 40), v18);
    if ( v22[0] != 6 )
    {
      v13 = sub_22C1480(a1, v19);
      sub_22CDEF0((__int64)v23, v13, v8, *(_QWORD *)(v18 + 40), v18);
      v14 = *(_QWORD ***)(v7 + 8);
      v15 = *((unsigned __int8 *)v14 + 8);
      if ( (unsigned int)(v15 - 17) > 1 )
      {
        v17 = sub_BCB2A0(*v14);
      }
      else
      {
        BYTE4(v21) = (_BYTE)v15 == 18;
        LODWORD(v21) = *((_DWORD *)v14 + 8);
        v16 = (__int64 *)sub_BCB2A0(*v14);
        v17 = sub_BCE1B0(v16, v21);
      }
      v11 = sub_22EAB60(v22, a2, v17, v23, v19 + 312);
      sub_22C0090(v23);
    }
    sub_22C0090(v22);
  }
  return v11;
}
