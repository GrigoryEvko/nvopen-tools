// Function: sub_73D220
// Address: 0x73d220
//
__int64 sub_73D220()
{
  __int64 v0; // r12
  __int64 v2; // rax
  __int64 v3; // r12
  char v4; // al
  __int64 v5; // rax
  __int64 v6; // r12
  char v7; // al
  __int64 v8; // rcx
  _QWORD *v9; // rax
  int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rax
  int v15; // ecx
  char v16; // dl
  __int64 v17; // r14
  __int64 v18; // r8
  __int64 v19; // r15
  __int64 v20; // rbx
  const __m128i *v21; // rax
  __m128i *v22; // rax
  __int64 v23; // r13
  const __m128i *v24; // rax
  __m128i *v25; // rax
  const __m128i *v26; // rax
  __m128i *v27; // rax
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-38h]

  v0 = qword_4F07AA0;
  if ( !qword_4F07AA0 )
  {
    v2 = sub_8865A0("source_location");
    v3 = v2;
    if ( v2 )
    {
      v4 = *(_BYTE *)(v2 + 80);
      if ( (unsigned __int8)(v4 - 4) <= 1u || v4 == 3 && (unsigned int)sub_8D3A70(*(_QWORD *)(v3 + 88)) )
      {
        if ( *(_QWORD *)(v3 + 88) )
        {
          v5 = sub_879C70("__impl");
          v6 = v5;
          if ( v5 )
          {
            v7 = *(_BYTE *)(v5 + 80);
            if ( (unsigned __int8)(v7 - 4) <= 1u || v7 == 3 && (unsigned int)sub_8D3A70(*(_QWORD *)(v6 + 88)) )
            {
              v0 = *(_QWORD *)(v6 + 88);
              v8 = *(_QWORD *)(v0 + 168);
              v9 = *(_QWORD **)v8;
              if ( !*(_QWORD *)v8 )
                goto LABEL_26;
              v10 = 0;
              do
              {
                v9 = (_QWORD *)*v9;
                ++v10;
              }
              while ( v9 );
              if ( !v10 )
              {
LABEL_26:
                v11 = *(_QWORD *)(v8 + 152);
                v12 = *(_QWORD *)(v11 + 112);
                if ( !v12 )
                  goto LABEL_44;
                v13 = 0;
                do
                {
                  v12 = *(_QWORD *)(v12 + 112);
                  ++v13;
                }
                while ( v12 );
                if ( !v13 )
                {
LABEL_44:
                  v14 = *(_QWORD *)(v11 + 144);
                  if ( !v14 )
                    goto LABEL_27;
                  v15 = 0;
                  do
                  {
                    v16 = *(_BYTE *)(v14 + 193);
                    v14 = *(_QWORD *)(v14 + 112);
                    v15 += (v16 & 0x10) == 0;
                  }
                  while ( v14 );
                  if ( !v15 )
                  {
LABEL_27:
                    v17 = *(_QWORD *)(v0 + 160);
                    if ( v17 )
                    {
                      v18 = *(_QWORD *)(v17 + 112);
                      if ( v18 )
                      {
                        v19 = *(_QWORD *)(v18 + 112);
                        if ( v19 )
                          v20 = *(_QWORD *)(v19 + 112);
                        else
                          v20 = 0;
                        v29 = *(_QWORD *)(v17 + 112);
                        v21 = (const __m128i *)sub_72BA30(0);
                        v22 = sub_73C570(v21, 1);
                        v23 = sub_72D2E0(v22);
                        if ( (unsigned int)sub_728C90(v17, v23)
                          && (unsigned int)sub_728C90(v29, v23)
                          && v19
                          && (unsigned int)sub_728C90(v19, v23)
                          && v20
                          && (unsigned int)sub_728C90(v20, v23)
                          && !*(_QWORD *)(v20 + 112) )
                        {
                          goto LABEL_8;
                        }
                      }
                      else
                      {
                        v26 = (const __m128i *)sub_72BA30(0);
                        v27 = sub_73C570(v26, 1);
                        v28 = sub_72D2E0(v27);
                        sub_728C90(v17, v28);
                      }
                    }
                    else
                    {
                      v24 = (const __m128i *)sub_72BA30(0);
                      v25 = sub_73C570(v24, 1);
                      sub_72D2E0(v25);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    v0 = qword_4F07AA0;
    if ( qword_4F07AA0 )
      return v0;
    sub_6851C0(0xCB4u, dword_4F07508);
    v0 = sub_72C930();
LABEL_8:
    qword_4F07AA0 = v0;
  }
  return v0;
}
