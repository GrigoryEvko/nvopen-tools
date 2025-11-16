// Function: sub_732700
// Address: 0x732700
//
__int64 __fastcall sub_732700(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // r13
  _QWORD *v15; // rax
  _QWORD *v16; // r13
  _QWORD *v17; // rax
  _QWORD *v18; // r13
  _QWORD *v19; // rax
  _QWORD *v20; // r13
  _QWORD *v21; // rax
  _QWORD *v22; // r13
  _QWORD *v23; // rax
  _QWORD *v25; // [rsp+0h] [rbp-50h]

  v9 = sub_7259C0(7);
  v9[20] = a1;
  v10 = v9[21];
  v11 = (__int64)v9;
  if ( a2 )
  {
    v12 = sub_72B0C0(a2, &dword_4F077C8);
    *(_QWORD *)v10 = v12;
    *((_DWORD *)v12 + 9) = 1;
    if ( a3 )
    {
      v25 = *(_QWORD **)v10;
      v13 = sub_72B0C0(a3, &dword_4F077C8);
      *v25 = v13;
      *((_DWORD *)v13 + 9) = 2;
      if ( a4 )
      {
        v14 = (_QWORD *)*v25;
        v15 = sub_72B0C0(a4, &dword_4F077C8);
        *v14 = v15;
        *((_DWORD *)v15 + 9) = 3;
        if ( a5 )
        {
          v16 = (_QWORD *)*v14;
          v17 = sub_72B0C0(a5, &dword_4F077C8);
          *v16 = v17;
          *((_DWORD *)v17 + 9) = 4;
          if ( a6 )
          {
            v18 = (_QWORD *)*v16;
            v19 = sub_72B0C0(a6, &dword_4F077C8);
            *v18 = v19;
            *((_DWORD *)v19 + 9) = 5;
            if ( a7 )
            {
              v20 = (_QWORD *)*v18;
              v21 = sub_72B0C0(a7, &dword_4F077C8);
              *v20 = v21;
              *((_DWORD *)v21 + 9) = 6;
              if ( a8 )
              {
                v22 = (_QWORD *)*v20;
                v23 = sub_72B0C0(a8, &dword_4F077C8);
                *v22 = v23;
                *((_DWORD *)v23 + 9) = 7;
              }
            }
          }
        }
      }
    }
  }
  *(_BYTE *)(v10 + 16) |= 2u;
  sub_7325D0(v11, &dword_4F077C8);
  return v11;
}
