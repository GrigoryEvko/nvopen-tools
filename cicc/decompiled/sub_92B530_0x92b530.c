// Function: sub_92B530
// Address: 0x92b530
//
__int64 __fastcall sub_92B530(unsigned int **a1, unsigned int a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  int v5; // r15d
  __int64 (*v8)(void); // rax
  __int64 v9; // r12
  _QWORD *v11; // rdx
  int v12; // ecx
  __int64 v13; // rax
  int v14; // esi
  unsigned int *v15; // rbx
  __int64 v16; // r13
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v20; // [rsp+18h] [rbp-68h]
  char v21[32]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  v5 = (int)a4;
  v8 = *(__int64 (**)(void))(*(_QWORD *)a1[10] + 56LL);
  if ( (char *)v8 != (char *)sub_928890 )
  {
    v9 = v8();
LABEL_5:
    if ( v9 )
      return v9;
    goto LABEL_7;
  }
  if ( *(_BYTE *)a3 <= 0x15u && *a4 <= 0x15u )
  {
    v9 = sub_AAB310(a2, a3, a4);
    goto LABEL_5;
  }
LABEL_7:
  v22 = 257;
  v9 = sub_BD2C40(72, unk_3F10FD0);
  if ( v9 )
  {
    v11 = *(_QWORD **)(a3 + 8);
    v12 = *((unsigned __int8 *)v11 + 8);
    if ( (unsigned int)(v12 - 17) > 1 )
    {
      v14 = sub_BCB2A0(*v11);
    }
    else
    {
      BYTE4(v20) = (_BYTE)v12 == 18;
      LODWORD(v20) = *((_DWORD *)v11 + 8);
      v13 = sub_BCB2A0(*v11);
      v14 = sub_BCE1B0(v13, v20);
    }
    sub_B523C0(v9, v14, 53, a2, a3, v5, (__int64)v21, 0, 0, 0);
  }
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v9,
    a5,
    a1[7],
    a1[8]);
  v15 = *a1;
  v16 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( *a1 != (unsigned int *)v16 )
  {
    do
    {
      v17 = *((_QWORD *)v15 + 1);
      v18 = *v15;
      v15 += 4;
      sub_B99FD0(v9, v18, v17);
    }
    while ( (unsigned int *)v16 != v15 );
  }
  return v9;
}
