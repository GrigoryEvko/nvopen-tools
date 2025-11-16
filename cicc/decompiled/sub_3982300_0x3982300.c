// Function: sub_3982300
// Address: 0x3982300
//
unsigned __int64 __fastcall sub_3982300(__int64 *a1, __int64 a2, __int16 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdi
  void (*v9)(); // rcx
  __int64 *v10; // rax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int16 v13; // dx
  bool v14; // r13
  unsigned int v15; // eax
  _QWORD *v17; // [rsp+0h] [rbp-50h] BYREF
  __int64 v18; // [rsp+8h] [rbp-48h]
  _QWORD v19[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v20; // [rsp+20h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 416) )
  {
    v8 = *(_QWORD *)(a2 + 256);
    v9 = *(void (**)())(*(_QWORD *)v8 + 104LL);
    if ( (*(_BYTE *)*a1 & 4) != 0 )
    {
      v10 = *(__int64 **)(*a1 - 8);
      v11 = *v10;
      v12 = v10 + 2;
    }
    else
    {
      v11 = 0;
      v12 = 0;
    }
    v17 = v12;
    v20 = 1283;
    v19[0] = "Label: ";
    v18 = v11;
    v19[1] = &v17;
    if ( v9 != nullsub_580 )
      ((void (__fastcall *)(__int64, _QWORD *, __int64, void (*)(), __int64, __int64, _QWORD *, __int64))v9)(
        v8,
        v19,
        1,
        v9,
        a5,
        a6,
        v17,
        v18);
  }
  if ( (unsigned __int16)a3 > 0x17u )
  {
    v14 = 1;
    v13 = a3;
  }
  else
  {
    v13 = a3;
    v14 = (((unsigned __int64)&loc_814040 >> a3) & 1) == 0;
  }
  v15 = sub_39822D0((__int64)a1, a2, v13);
  return sub_396F390(a2, *a1, 0, v15, !v14);
}
