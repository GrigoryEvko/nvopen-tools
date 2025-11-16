// Function: sub_1909530
// Address: 0x1909530
//
void __fastcall sub_1909530(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  size_t v5; // rdx
  __int64 *v6; // r14
  char *v7; // r15
  __int64 v8; // rax
  size_t v9; // rdx
  __int64 *v10; // r14
  char *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r14
  __int64 v15; // rax
  _BYTE *v16; // r15
  __int64 v17; // rdi
  unsigned int v18; // r14d
  __int64 v19; // rsi
  __int64 v20; // kr00_8
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int v30; // [rsp-E8h] [rbp-E8h]
  unsigned int v31; // [rsp-E4h] [rbp-E4h]
  _BYTE *v32; // [rsp-E0h] [rbp-E0h]
  unsigned int v33[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v34; // [rsp-C0h] [rbp-C0h]
  __int64 v35; // [rsp-B8h] [rbp-B8h]
  __int64 v36; // [rsp-B0h] [rbp-B0h]
  int v37; // [rsp-A8h] [rbp-A8h]
  _BYTE *v38; // [rsp-88h] [rbp-88h] BYREF
  __int64 v39; // [rsp-80h] [rbp-80h]
  _BYTE v40[120]; // [rsp-78h] [rbp-78h] BYREF

  if ( *(_BYTE *)(a2 + 16) > 0x17u )
  {
    if ( *(_BYTE *)(a1 + 16) != 54 )
      sub_15F2780((unsigned __int8 *)a2, a1);
    v4 = sub_16498A0(a1);
    v5 = 0;
    v6 = (__int64 *)v4;
    v7 = off_4CD4978[0];
    if ( off_4CD4978[0] )
      v5 = strlen(off_4CD4978[0]);
    v31 = sub_1602B80(v6, v7, v5);
    v8 = sub_16498A0(a1);
    v9 = 0;
    v10 = (__int64 *)v8;
    v11 = off_4CD4970[0];
    if ( off_4CD4970[0] )
      v9 = strlen(off_4CD4970[0]);
    v30 = sub_1602B80(v10, v11, v9);
    v33[1] = v30;
    v34 = 0x700000001LL;
    v35 = 0x400000008LL;
    v36 = 0x600000003LL;
    v38 = v40;
    v33[0] = v31;
    v37 = 16;
    v39 = 0x400000000LL;
    sub_1624960(a2, v33, 9);
    if ( *(__int16 *)(a2 + 18) < 0 )
      sub_161F980(a2, (__int64)&v38);
    v13 = (__int64)v38;
    v14 = 16LL * (unsigned int)v39;
    v15 = (__int64)&v38[v14];
    v32 = &v38[v14];
    if ( v13 != v13 + v14 )
    {
      v16 = v38;
      while ( 1 )
      {
        v17 = *(_QWORD *)(a1 + 48);
        v18 = *(_DWORD *)v16;
        if ( v17 )
        {
LABEL_13:
          v15 = sub_1625790(a1, v18);
          v17 = v15;
          goto LABEL_14;
        }
        while ( 2 )
        {
          if ( *(__int16 *)(a1 + 18) < 0 )
            goto LABEL_13;
LABEL_14:
          v19 = *((_QWORD *)v16 + 1);
          v20 = v15;
          v15 = v18;
          switch ( v18 )
          {
            case 1u:
              v29 = sub_14A8140(v17, v19);
              sub_1625C10(a2, 1, v29);
              goto LABEL_27;
            case 3u:
              v28 = sub_161F2A0(v17, v19, v12, v13);
              sub_1625C10(a2, 3, v28);
              goto LABEL_27;
            case 4u:
              v27 = sub_1628300(v17, v19);
              sub_1625C10(a2, 4, v27);
              goto LABEL_27;
            case 6u:
              sub_1625C10(a2, 6, v17);
              goto LABEL_27;
            case 7u:
              v25 = sub_1631A90(v17, v19);
              sub_1625C10(a2, 7, v25);
              goto LABEL_27;
            case 8u:
            case 0xAu:
              v24 = sub_1630FC0(v17, v19);
              sub_1625C10(a2, v18, v24);
              goto LABEL_27;
            case 0xBu:
              sub_1625C10(a2, 11, v17);
              goto LABEL_27;
            case 0xCu:
            case 0xDu:
              v23 = sub_161F460(v17, v19);
              sub_1625C10(a2, v18, v23);
              goto LABEL_27;
            case 0x10u:
              goto LABEL_27;
            case 0x11u:
              v26 = sub_161F460(v17, v19);
              sub_1625C10(a2, 17, v26);
              goto LABEL_27;
            default:
              v15 = v20;
              if ( v31 == v18 )
              {
                if ( v19 != v17 )
                  goto LABEL_19;
                goto LABEL_27;
              }
              if ( v30 == v18 )
              {
                if ( v17 )
                {
                  v12 = 1LL - *(unsigned int *)(v19 + 8);
                  v13 = *(unsigned int *)(v17 + 8);
                  v15 = *(_QWORD *)(v17 + 8 * (1 - v13));
                  if ( *(_QWORD *)(v19 + 8 * v12) == v15 )
                  {
LABEL_27:
                    v16 += 16;
                    if ( v32 == v16 )
                      goto LABEL_20;
                    v17 = *(_QWORD *)(a1 + 48);
                    v18 = *(_DWORD *)v16;
                    if ( v17 )
                      goto LABEL_13;
                    continue;
                  }
                }
              }
LABEL_19:
              v16 += 16;
              sub_1625C10(a2, v18, 0);
              if ( v32 == v16 )
                goto LABEL_20;
              break;
          }
          break;
        }
      }
    }
LABEL_20:
    if ( *(_QWORD *)(a1 + 48) || *(__int16 *)(a1 + 18) < 0 )
    {
      v21 = sub_1625790(a1, 16);
      if ( v21 && (unsigned __int8)(*(_BYTE *)(a2 + 16) - 54) <= 1u )
      {
        sub_1625C10(a2, 16, v21);
        v22 = (unsigned __int64)v38;
        if ( v38 == v40 )
          return;
      }
      else
      {
        v22 = (unsigned __int64)v38;
        if ( v38 == v40 )
          return;
      }
    }
    else
    {
      v22 = (unsigned __int64)v38;
      if ( v38 == v40 )
        return;
    }
    _libc_free(v22);
  }
}
