// Function: sub_35E5E40
// Address: 0x35e5e40
//
void __fastcall sub_35E5E40(_DWORD *a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // rbx
  _DWORD *v6; // r10
  _DWORD *v7; // r9
  signed __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r9
  __int64 v11; // r10
  _DWORD *v12; // r11
  _DWORD *v13; // r15
  __int64 v14; // r14
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  int v17; // ecx
  int v18; // ecx
  int v19; // ecx
  unsigned __int64 v21; // [rsp+8h] [rbp-48h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  _DWORD *v23; // [rsp+18h] [rbp-38h]

  if ( a5 )
  {
    v5 = a4;
    if ( a4 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a5;
      if ( a4 + a5 == 2 )
      {
        v15 = (unsigned __int64)a1;
        v13 = a2;
LABEL_12:
        if ( *v13 < *(_DWORD *)v15 )
        {
          v16 = *(_QWORD *)(v15 + 16);
          *(_QWORD *)(v15 + 16) = *((_QWORD *)v13 + 2);
          v17 = v13[2];
          *((_QWORD *)v13 + 2) = v16;
          LODWORD(v16) = *(_DWORD *)(v15 + 8);
          *(_DWORD *)(v15 + 8) = v17;
          v18 = v13[1];
          v13[2] = v16;
          LODWORD(v16) = *(_DWORD *)(v15 + 4);
          *(_DWORD *)(v15 + 4) = v18;
          v19 = *v13;
          v13[1] = v16;
          LODWORD(v16) = *(_DWORD *)v15;
          *(_DWORD *)v15 = v19;
          *v13 = v16;
        }
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v9 = v5 / 2;
        v13 = sub_35E5080(v7, a3, &v6[2 * (v5 / 2) + 2 * ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL)]);
        v14 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v13 - v10) >> 3);
        while ( 1 )
        {
          v22 = v11;
          v23 = v12;
          v8 -= v14;
          v21 = sub_35E4E60((__int64)v12, v10, (__int64)v13);
          sub_35E5E40(v22, v23, v21, v9, v14);
          v5 -= v9;
          if ( !v5 )
            break;
          v15 = v21;
          if ( !v8 )
            break;
          if ( v8 + v5 == 2 )
            goto LABEL_12;
          v6 = (_DWORD *)v21;
          v7 = v13;
          if ( v8 < v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v8 / 2;
          v13 = &v7[2 * (v8 / 2) + 2 * ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
          v12 = sub_35E50E0(v6, (__int64)v7, v13);
          v9 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v12 - v11) >> 3);
        }
      }
    }
  }
}
