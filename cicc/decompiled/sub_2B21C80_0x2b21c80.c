// Function: sub_2B21C80
// Address: 0x2b21c80
//
void __fastcall sub_2B21C80(__int64 a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v4; // rax
  __int64 *v5; // r12
  __int64 v6; // rcx
  unsigned __int8 ****v7; // r14
  __int64 v8; // r10
  char v9; // r9
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // r13d
  unsigned __int8 *v13; // rbx
  unsigned __int8 **v14; // rdi
  __int64 v15; // rsi
  _BYTE **v16; // rax
  __int64 v17; // [rsp+8h] [rbp-58h]
  const char *v18; // [rsp+10h] [rbp-50h] BYREF
  char v19; // [rsp+30h] [rbp-30h]
  char v20; // [rsp+31h] [rbp-2Fh]

  v4 = sub_2B21190(*(_QWORD *)(a1 + 24), *(_QWORD *)a1, a2, a3, a4, **(_QWORD **)(a1 + 16));
  v5 = *(__int64 **)(a1 + 32);
  v6 = v4;
  if ( *v5 )
  {
    v7 = *(unsigned __int8 *****)(a1 + 24);
    v8 = *(_QWORD *)a1;
    v20 = 1;
    v18 = "op.rdx";
    v9 = 1;
    v19 = 3;
    v10 = *((unsigned int *)v7 + 2);
    v11 = *v5;
    v12 = *((_DWORD *)v7 + 394);
    if ( v10 != 2 )
    {
      v9 = 0;
      if ( v10 == 1 )
      {
        v17 = *v5;
        v14 = **v7;
        v15 = (__int64)&v14[*((unsigned int *)*v7 + 2)];
        v16 = sub_2B0AAE0(v14, v15);
        v11 = v17;
        v9 = v15 != (_QWORD)v16;
      }
    }
    v13 = (unsigned __int8 *)sub_2B21610(v8, v12, v11, v6, (__int64)&v18, v9);
    if ( v12 - 6 <= 3 && *v13 == 86 )
    {
      sub_F70480(*((unsigned __int8 **)v13 - 12), **v7, *((unsigned int *)*v7 + 2), 0, 0);
      sub_F70480(v13, (*v7)[18], *((unsigned int *)*v7 + 38), 0, 0);
    }
    else
    {
      sub_F70480(v13, **v7, *((unsigned int *)*v7 + 2), 0, 0);
    }
    *v5 = (__int64)v13;
  }
  else
  {
    *v5 = v4;
  }
}
