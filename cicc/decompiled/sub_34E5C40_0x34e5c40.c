// Function: sub_34E5C40
// Address: 0x34e5c40
//
bool __fastcall sub_34E5C40(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  _QWORD *v7; // rax
  unsigned int v8; // edx
  int v9; // ebx
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  __int64 v13; // rax
  __int64 v14; // rdi
  __int16 v15; // dx
  unsigned __int64 v16; // r12
  void *v17; // r8
  size_t v18; // r11
  __int64 *v19; // rdx
  int v20; // r11d
  __int64 *v21; // r12
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 *v24; // rdi
  __int64 v25; // [rsp+10h] [rbp-90h]
  __int64 *src; // [rsp+18h] [rbp-88h]
  void *srca; // [rsp+18h] [rbp-88h]
  __int64 *v28; // [rsp+20h] [rbp-80h] BYREF
  __int64 v29; // [rsp+28h] [rbp-78h]
  _BYTE v30[112]; // [rsp+30h] [rbp-70h] BYREF

  v6 = a2[41];
  if ( a2 + 40 != (_QWORD *)v6 )
  {
    v7 = (_QWORD *)a2[41];
    v8 = 0;
    do
    {
      v7 = (_QWORD *)v7[1];
      ++v8;
    }
    while ( a2 + 40 != v7 );
    if ( v8 > 1 )
    {
      v9 = 0;
      v25 = a2[8];
      v10 = a2[40] & 0xFFFFFFFFFFFFFFF8LL;
      if ( v10 != v6 )
      {
        while ( 1 )
        {
          v11 = v6;
          v6 = *(_QWORD *)(v6 + 8);
          if ( *(_BYTE *)(v11 + 216) || *(_BYTE *)(v11 + 217) || *(_QWORD *)(v11 + 224) )
            goto LABEL_9;
          v13 = *(_QWORD *)(v11 + 56);
          v14 = v11 + 48;
          if ( v11 + 48 != v13 )
          {
            while ( 1 )
            {
              v15 = *(_WORD *)(v13 + 68);
              if ( v15 != 10 && (unsigned __int16)(v15 - 3) > 4u && (unsigned __int16)(v15 - 14) > 4u )
                break;
              if ( (*(_BYTE *)v13 & 4) != 0 )
              {
                v13 = *(_QWORD *)(v13 + 8);
                if ( v14 == v13 )
                  goto LABEL_18;
              }
              else
              {
                while ( (*(_BYTE *)(v13 + 44) & 8) != 0 )
                  v13 = *(_QWORD *)(v13 + 8);
                v13 = *(_QWORD *)(v13 + 8);
                if ( v14 == v13 )
                  goto LABEL_18;
              }
            }
            if ( v14 != v13 )
              goto LABEL_9;
          }
LABEL_18:
          v16 = *(unsigned int *)(v11 + 72);
          v17 = *(void **)(v11 + 64);
          v28 = (__int64 *)v30;
          v18 = 8 * v16;
          v29 = 0x800000000LL;
          if ( v16 > 8 )
            break;
          v19 = (__int64 *)v30;
          if ( v18 )
          {
            v24 = (__int64 *)v30;
            goto LABEL_35;
          }
LABEL_20:
          v20 = v16 + v18;
          v21 = v19;
          LODWORD(v29) = v20;
          src = &v19[v20];
          if ( src != v19 )
          {
            do
            {
              v22 = *v21++;
              sub_2E337A0(v22, v11, v6);
            }
            while ( src != v21 );
          }
          if ( v25 )
            sub_2E79DC0(v25, v11, v6);
          while ( 1 )
          {
            v23 = *(_DWORD *)(v11 + 120);
            if ( !v23 )
              break;
            sub_2E33590(v11, (__int64 *)(*(_QWORD *)(v11 + 112) + 8LL * v23 - 8), 0);
          }
          ++v9;
          sub_2E32710((_QWORD *)v11);
          if ( v28 != (__int64 *)v30 )
            _libc_free((unsigned __int64)v28);
LABEL_9:
          if ( v10 == v6 )
            return v9 != 0;
        }
        srca = v17;
        sub_C8D5F0((__int64)&v28, v30, v16, 8u, (__int64)v17, a6);
        v17 = srca;
        v18 = 8 * v16;
        v24 = &v28[(unsigned int)v29];
LABEL_35:
        memcpy(v24, v17, v18);
        v19 = v28;
        LODWORD(v18) = v29;
        goto LABEL_20;
      }
    }
  }
  return 0;
}
