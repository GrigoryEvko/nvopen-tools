// Function: sub_3380BE0
// Address: 0x3380be0
//
void __fastcall sub_3380BE0(__int64 a1, unsigned __int8 *a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  int v7; // ecx
  bool v9; // bl
  __int64 *v10; // rdx
  __int64 v11; // rcx
  int v12; // eax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 *v15; // rax
  __int64 v16; // rcx
  __int64 *v17; // [rsp+0h] [rbp-40h]
  __int64 *v18; // [rsp+0h] [rbp-40h]
  __int64 v19[7]; // [rsp+8h] [rbp-38h] BYREF

  v19[0] = (__int64)a2;
  if ( a2 )
  {
    v7 = *a2;
    if ( (unsigned int)(v7 - 12) > 1 && (*((_QWORD *)a2 + 2) || (_BYTE)v7 == 22) )
    {
      v9 = (_BYTE)v7 == 22 || *(_WORD *)(a3 + 20) != 0;
      v10 = sub_337DC20(a1 + 8, v19);
      if ( !*v10 )
      {
        if ( *(_BYTE *)v19[0] != 22 )
          goto LABEL_17;
        v18 = v10;
        v15 = sub_337DC20(a1 + 40, v19);
        v10 = v18;
        v16 = *v15;
        *v18 = *v15;
        *((_DWORD *)v18 + 2) = *((_DWORD *)v15 + 2);
        if ( !v16 )
          goto LABEL_17;
      }
      if ( *(_BYTE *)v19[0] == 78 )
        v19[0] = *(_QWORD *)(v19[0] - 32);
      v11 = *v10;
      v12 = *(_DWORD *)(*v10 + 24);
      if ( (v12 == 39 || v12 == 15) && v9 )
      {
        v13 = sub_33E6410(*(_QWORD *)(a1 + 864), a3, (_DWORD)a4, *(_DWORD *)(v11 + 96), 1, a5, *(_DWORD *)(a1 + 848));
      }
      else
      {
        if ( *(_BYTE *)v19[0] == 22 )
        {
LABEL_17:
          v17 = v10;
          v14 = sub_B10CD0(a5);
          sub_337F9F0(a1, v19[0], a3, a4, v14, 1, v17);
          return;
        }
        v13 = sub_33E60C0(
                *(_QWORD *)(a1 + 864),
                a3,
                (_DWORD)a4,
                v11,
                *((_DWORD *)v10 + 2),
                1,
                a5,
                *(_DWORD *)(a1 + 848));
      }
      sub_33F99B0(*(_QWORD *)(a1 + 864), v13, v9);
    }
  }
}
