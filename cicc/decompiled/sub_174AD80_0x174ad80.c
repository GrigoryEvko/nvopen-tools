// Function: sub_174AD80
// Address: 0x174ad80
//
__int64 __fastcall sub_174AD80(__int64 a1, __int64 a2, int *a3, __int64 *a4)
{
  unsigned int v4; // r13d
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r9d
  __int64 v12; // rax
  _QWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rbx
  char v17; // al
  __int64 v18; // rdx
  _BYTE *v19; // rdi
  __int64 v20; // rdx
  _BYTE *v21; // r9
  _QWORD *v22; // rax
  _QWORD *v23; // r14
  unsigned int v24; // r14d
  unsigned int v25; // eax
  char v26; // al
  __int64 v27; // rdx
  _BYTE *v28; // rdi
  __int64 v29; // rdx
  _BYTE *v30; // r9
  _QWORD *v31; // rbx
  _QWORD *v32; // rdx
  int v33; // eax
  _QWORD *v34; // rax
  unsigned int v35; // r13d
  __int64 v37; // rdi
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned int v40; // eax
  unsigned int v41; // r14d
  __int64 v42; // [rsp-8h] [rbp-78h]
  unsigned __int8 v43; // [rsp+Bh] [rbp-65h]
  int v44; // [rsp+Ch] [rbp-64h]
  int *v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  __int64 v47; // [rsp+18h] [rbp-58h]
  _BYTE *v48; // [rsp+18h] [rbp-58h]
  _BYTE *v49; // [rsp+18h] [rbp-58h]
  int v50; // [rsp+18h] [rbp-58h]
  int v51; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v52[8]; // [rsp+30h] [rbp-40h] BYREF

  *a3 = 0;
  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
    return 1;
  v4 = sub_1749B70(a1, a2);
  if ( (_BYTE)v4 )
  {
    return 1;
  }
  else if ( (unsigned __int8)v11 > 0x17u )
  {
    v12 = *(_QWORD *)(a1 + 8);
    if ( v12 )
    {
      if ( !*(_QWORD *)(v12 + 8) )
      {
        switch ( v11 )
        {
          case '#':
          case '%':
          case '\'':
          case '2':
          case '3':
          case '4':
            if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              v13 = *(_QWORD **)(a1 - 8);
            else
              v13 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
            v47 = v10;
            if ( !(unsigned __int8)sub_174AD80(*v13, a2, a3, a4) )
              return v4;
            v14 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
                ? *(_QWORD *)(a1 - 8)
                : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
            if ( !(unsigned __int8)sub_174AD80(*(_QWORD *)(v14 + 24), a2, &v51, a4) )
              return v4;
            if ( !*a3 )
            {
              LOBYTE(v4) = v51 == 0;
              return v4;
            }
            if ( v51 )
              return v4;
            if ( (unsigned int)*(unsigned __int8 *)(a1 + 16) - 50 > 2 )
              return v4;
            v46 = v47;
            v50 = *a3;
            v40 = sub_16431D0(*(_QWORD *)a1);
            sub_171A350((__int64)v52, v40, v50);
            v42 = sub_13CF970(a1);
            v41 = sub_14C1670(*(_QWORD *)(v42 + 24), (__int64)v52, a4[333], 0, a4[330], v46, a4[332]);
            sub_135E100(v52);
            if ( !(_BYTE)v41 )
              return v4;
            if ( *(_BYTE *)(a1 + 16) != 50 )
              return 1;
            *a3 = 0;
            return v41;
          case '/':
            v26 = *(_BYTE *)(a1 + 23) & 0x40;
            if ( v26 )
              v27 = *(_QWORD *)(a1 - 8);
            else
              v27 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
            v28 = *(_BYTE **)(v27 + 24);
            v29 = (unsigned __int8)v28[16];
            v30 = v28 + 24;
            if ( (_BYTE)v29 == 13 )
              goto LABEL_37;
            if ( *(_BYTE *)(*(_QWORD *)v28 + 8LL) == 16 && (unsigned __int8)v29 <= 0x10u )
            {
              v39 = sub_15A1020(v28, a2, v29, v9);
              if ( v39 )
              {
                if ( *(_BYTE *)(v39 + 16) == 13 )
                {
                  v30 = (_BYTE *)(v39 + 24);
                  v26 = *(_BYTE *)(a1 + 23) & 0x40;
LABEL_37:
                  if ( v26 )
                    v31 = *(_QWORD **)(a1 - 8);
                  else
                    v31 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
                  v49 = v30;
                  v4 = sub_174AD80(*v31, a2, a3, a4);
                  if ( (_BYTE)v4 )
                  {
                    v32 = *(_QWORD **)v49;
                    if ( *((_DWORD *)v49 + 2) > 0x40u )
                      v32 = (_QWORD *)*v32;
                    v33 = *a3 - (_DWORD)v32;
                    if ( (unsigned int)*a3 <= (unsigned __int64)v32 )
                      v33 = 0;
                    *a3 = v33;
                  }
                }
              }
            }
            return v4;
          case '0':
            v17 = *(_BYTE *)(a1 + 23) & 0x40;
            if ( v17 )
              v18 = *(_QWORD *)(a1 - 8);
            else
              v18 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
            v19 = *(_BYTE **)(v18 + 24);
            v20 = (unsigned __int8)v19[16];
            v21 = v19 + 24;
            if ( (_BYTE)v20 == 13 )
              goto LABEL_27;
            if ( *(_BYTE *)(*(_QWORD *)v19 + 8LL) == 16 && (unsigned __int8)v20 <= 0x10u )
            {
              v38 = sub_15A1020(v19, a2, v20, v9);
              if ( v38 )
              {
                if ( *(_BYTE *)(v38 + 16) == 13 )
                {
                  v21 = (_BYTE *)(v38 + 24);
                  v17 = *(_BYTE *)(a1 + 23) & 0x40;
LABEL_27:
                  if ( v17 )
                    v22 = *(_QWORD **)(a1 - 8);
                  else
                    v22 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
                  v48 = v21;
                  v4 = sub_174AD80(*v22, a2, a3, a4);
                  if ( (_BYTE)v4 )
                  {
                    v23 = *(_QWORD **)v48;
                    if ( *((_DWORD *)v48 + 2) > 0x40u )
                      v23 = **(_QWORD ***)v48;
                    v24 = *a3 + (_DWORD)v23;
                    *a3 = v24;
                    v25 = sub_16431D0(*(_QWORD *)a1);
                    if ( v24 > v25 )
                      *a3 = v25;
                  }
                }
              }
            }
            return v4;
          case '<':
          case '=':
          case '>':
            return 1;
          case 'M':
            if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              v34 = *(_QWORD **)(a1 - 8);
            else
              v34 = (_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
            if ( !(unsigned __int8)sub_174AD80(*v34, a2, a3, a4) )
              return v4;
            v44 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
            if ( v44 == 1 )
              return 1;
            v43 = v4;
            v35 = 1;
            v45 = a3;
            break;
          case 'O':
            if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
              v15 = *(_QWORD *)(a1 - 8);
            else
              v15 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
            if ( (unsigned __int8)sub_174AD80(*(_QWORD *)(v15 + 24), a2, &v51, a4) )
            {
              v16 = (*(_BYTE *)(a1 + 23) & 0x40) != 0
                  ? *(_QWORD *)(a1 - 8)
                  : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
              if ( (unsigned __int8)sub_174AD80(*(_QWORD *)(v16 + 48), a2, a3, a4) )
                LOBYTE(v4) = *a3 == v51;
            }
            return v4;
          default:
            return v4;
        }
        while ( 1 )
        {
          v37 = (*(_BYTE *)(a1 + 23) & 0x40) != 0 ? *(_QWORD *)(a1 - 8) : a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
          if ( !(unsigned __int8)sub_174AD80(*(_QWORD *)(v37 + 24LL * v35), a2, &v51, a4) || *v45 != v51 )
            break;
          if ( ++v35 == v44 )
            return 1;
        }
        return v43;
      }
    }
  }
  return v4;
}
