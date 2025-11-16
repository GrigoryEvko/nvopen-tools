// Function: .sigaction
// Address: 0x4069b0
//
// attributes: thunk
int sigaction(int sig, const struct sigaction *act, struct sigaction *oact)
{
  return sigaction(sig, act, oact);
}
