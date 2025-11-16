// Function: sub_19E3400
// Address: 0x19e3400
//
__int64 __fastcall sub_19E3400(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // r15
  __int64 v7; // rdx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r8
  _QWORD *v12; // rbx
  __int64 v13; // r12
  int v14; // r11d
  _QWORD *v15; // r10
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  int v20; // eax
  char v21; // al
  __int64 v22; // rax
  _QWORD *v23; // rax
  int v24; // [rsp+4h] [rbp-5Ch]
  int v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  int v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  unsigned int v29; // [rsp+10h] [rbp-50h]
  __int64 v30; // [rsp+10h] [rbp-50h]
  unsigned int v31; // [rsp+18h] [rbp-48h]
  _QWORD *v32; // [rsp+18h] [rbp-48h]
  int v33; // [rsp+18h] [rbp-48h]
  _QWORD *v34; // [rsp+20h] [rbp-40h]
  _QWORD *v35; // [rsp+20h] [rbp-40h]
  __int64 v36; // [rsp+28h] [rbp-38h]
  unsigned int v37; // [rsp+28h] [rbp-38h]
  unsigned int v38; // [rsp+28h] [rbp-38h]
  unsigned int v39; // [rsp+28h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    LODWORD(v9) = *(_QWORD *)(*a2 + 16);
    if ( !(_DWORD)v9 )
    {
      v36 = *(_QWORD *)(a1 + 8);
      v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 32LL))(v6);
      v7 = v36;
      *(_QWORD *)(v6 + 16) = v9;
      v6 = *a2;
    }
    v10 = (unsigned int)(v4 - 1);
    v11 = (unsigned int)v10 & (unsigned int)v9;
    v12 = (_QWORD *)(v7 + 16 * v11);
    v13 = *v12;
    if ( *v12 != v6 )
    {
      v14 = 1;
      v15 = 0;
      while ( 1 )
      {
        if ( v6 != -8 && v6 != 0x7FFFFFFF0LL && v13 != 0x7FFFFFFF0LL && v13 != -8 )
        {
          v16 = *(_QWORD *)(v13 + 16);
          if ( !(_DWORD)v16 )
          {
            v25 = v14;
            v28 = v7;
            v31 = v11;
            v34 = v15;
            v37 = v10;
            v17 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v13 + 32LL))(
                    v13,
                    v16,
                    v7,
                    v10,
                    v11,
                    0x7FFFFFFF0LL);
            v14 = v25;
            v7 = v28;
            v11 = v31;
            v15 = v34;
            *(_QWORD *)(v13 + 16) = v17;
            v16 = v17;
            v10 = v37;
          }
          v18 = *(_QWORD *)(v6 + 16);
          if ( !(_DWORD)v18 )
          {
            v24 = v14;
            v26 = v7;
            v29 = v11;
            v32 = v15;
            v38 = v10;
            v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 32LL))(
                    v6,
                    v16,
                    v7,
                    v10,
                    v11,
                    0x7FFFFFFF0LL);
            v14 = v24;
            v7 = v26;
            v11 = v29;
            v15 = v32;
            *(_QWORD *)(v6 + 16) = v18;
            v10 = v38;
          }
          if ( v16 == v18 )
          {
            v19 = *(_DWORD *)(v6 + 12);
            if ( v19 == *(_DWORD *)(v13 + 12) )
            {
              if ( v19 > 0xFFFFFFFD )
                break;
              v20 = *(_DWORD *)(v6 + 8);
              if ( (unsigned int)(v20 - 11) <= 1 || v20 == *(_DWORD *)(v13 + 8) )
              {
                v27 = v14;
                v30 = v7;
                v33 = v11;
                v35 = v15;
                v39 = v10;
                v21 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v6 + 16LL))(
                        v6,
                        v13,
                        v7,
                        v10,
                        v11,
                        0x7FFFFFFF0LL);
                v10 = v39;
                v15 = v35;
                LODWORD(v11) = v33;
                v7 = v30;
                v14 = v27;
                if ( v21 )
                  break;
              }
            }
          }
        }
        if ( *v12 == -8 )
        {
          if ( !v15 )
            v15 = v12;
          *a3 = v15;
          return 0;
        }
        if ( v15 || *v12 != 0x7FFFFFFF0LL )
          v12 = v15;
        v6 = *a2;
        v22 = (unsigned int)v10 & (v14 + (_DWORD)v11);
        v11 = v22;
        v23 = (_QWORD *)(v7 + 16 * v22);
        v13 = *v23;
        if ( *v23 == *a2 )
        {
          v12 = v23;
          break;
        }
        v15 = v12;
        ++v14;
        v12 = v23;
      }
    }
    *a3 = v12;
    return 1;
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}
